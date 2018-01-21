#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import h5py
import time
import logging
from models_img import CNNVAE
from optim_n2n import OptimN2N    
import utils
import torch.utils.data

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--data_file', default='data/omniglot/omniglot.pt')
parser.add_argument('--train_from', default='')
parser.add_argument('--checkpoint_path', default='baseline.pt')

# Model options
parser.add_argument('--img_size', default=[1,28,28])
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_layers', default=[64,64,64])
parser.add_argument('--dec_kernel_size', default=[9,9,9,7,7,7,5,5,5,3,3,3], type=int)
parser.add_argument('--dec_layers', default=[32,32,32,32,32,32,32,32,32,32,32,32])
parser.add_argument('--latent_feature_map', default=4, type=int)
parser.add_argument('--model', default='savi', type=str, choices = ['vae', 'autoreg', 'savi', 'svi'])
parser.add_argument('--train_kl', default=1, type=int)
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--acc_param_grads', default=1, type=int)

# Optimization options
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--svi_steps', default=20, type=int)
parser.add_argument('--svi_lr1', default=1, type=float)
parser.add_argument('--svi_lr2', default=1, type=float)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--warmup', default=10., type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--svi_max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=2, type=int)
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=500)
parser.add_argument('--test', type=int, default=0)
    
def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  all_data = torch.load(args.data_file)
  x_train, x_val, x_test = all_data
  y_size = 1 
  y_train = torch.zeros(x_train.size(0), y_size)
  y_val = torch.zeros(x_val.size(0), y_size)
  y_test = torch.zeros(x_test.size(0), y_size)
  train = torch.utils.data.TensorDataset(x_train, y_train)
  val = torch.utils.data.TensorDataset(x_val, y_val)
  test = torch.utils.data.TensorDataset(x_test, y_test)
  
  train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
  print('Train data: %d batches' % len(train_loader))
  print('Val data: %d batches' % len(val_loader))
  print('Test data: %d batches' % len(test_loader))
  if args.slurm == 0: 
    cuda.set_device(args.gpu)
  if args.model == 'autoreg':
    args.latent_feature_map = 0
  if args.train_from == '':
    model = CNNVAE(img_size = args.img_size,
                   latent_dim = args.latent_dim,
                   enc_layers = args.enc_layers,
                   dec_kernel_size = args.dec_kernel_size,
                   dec_layers = args.dec_layers,
                   latent_feature_map = args.latent_feature_map)
  else:
    print('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']    
  print("model architecture")
  print(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
  
  model.cuda()
  model.train()
    
  def variational_loss(input, img, model, z = None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)      
    preds = model._dec_forward(img, z_samples)
    nll = utils.log_bernoulli_loss(preds, img)
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + args.beta*kl

  update_params = list(model.dec.parameters())
  meta_optimizer = OptimN2N(variational_loss, model, update_params, eps = args.eps, 
                            lr = [args.svi_lr1, args.svi_lr2],
                            iters = args.svi_steps, momentum = args.momentum,
                            acc_param_grads= args.acc_param_grads == 1,  
                            max_grad_norm = args.svi_max_grad_norm)
  epoch = 0
  t = 0
  best_val_nll = 1e5
  best_epoch = 0
  loss_stats = []    
  if args.warmup == 0:
    args.beta = 1.
  else:
    args.beta = 0.1

  if args.test == 1:
    args.beta = 1
    eval(test_loader, model, meta_optimizer)
    exit()  
    
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1
    print('Starting epoch %d' % epoch)
    train_nll_vae = 0.
    train_nll_autoreg = 0.
    train_kl_vae = 0.
    train_nll_svi = 0.
    train_kl_svi = 0.
    num_examples = 0
    for b, datum in enumerate(train_loader):
      if args.warmup > 0:
        args.beta = min(1, args.beta + 1./(args.warmup*len(train_loader)))      
      img, _ = datum      
      img = torch.bernoulli(img)      
      batch_size = img.size(0)
      img = Variable(img.cuda())
      t += 1      
      optimizer.zero_grad()
      if args.model == 'autoreg':
        preds = model._dec_forward(img, None)
        nll_autoreg = utils.log_bernoulli_loss(preds, img)
        train_nll_autoreg += nll_autoreg.data[0]*batch_size
        nll_autoreg.backward()
      elif args.model == 'svi':
        mean_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        logvar_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img,
                                                t % args.print_every == 0)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
        preds = model._dec_forward(img, z_samples)
        nll_svi = utils.log_bernoulli_loss(preds, img)
        train_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        train_kl_svi += kl_svi.data[0]*batch_size      
        var_loss = nll_svi + args.beta*kl_svi
        var_loss.backward()
      else:
        mean, logvar = model._enc_forward(img)
        z_samples = model._reparameterize(mean, logvar)
        preds = model._dec_forward(img, z_samples)      
        nll_vae = utils.log_bernoulli_loss(preds, img)
        train_nll_vae += nll_vae.data[0]*batch_size
        kl_vae = utils.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.data[0]*batch_size        
        if args.model == 'vae':
          vae_loss = nll_vae + args.beta*kl_vae          
          vae_loss.backward(retain_graph = True)
          
        if args.model == 'savi':
          var_params = torch.cat([mean, logvar], 1)        
          mean_svi = Variable(mean.data, requires_grad = True)
          logvar_svi = Variable(logvar.data, requires_grad = True)
          
          var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img,
                                                  t % args.print_every == 0)
          mean_svi_final, logvar_svi_final = var_params_svi
          z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
          preds = model._dec_forward(img, z_samples)
          nll_svi = utils.log_bernoulli_loss(preds, img)
          train_nll_svi += nll_svi.data[0]*batch_size
          kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
          train_kl_svi += kl_svi.data[0]*batch_size      
          var_loss = nll_svi + args.beta*kl_svi
          var_loss.backward(retain_graph = True)          
          if args.train_n2n == 0:
            if args.train_kl == 1:
              mean_final = mean_svi_final.detach()
              logvar_final = logvar_svi_final.detach()            
              kl_init_final = utils.kl_loss(mean, logvar, mean_final, logvar_final)
              kl_init_final.backward(retain_graph = True)              
            else:
              vae_loss = nll_vae + args.beta*kl_vae
              var_param_grads = torch.autograd.grad(vae_loss, [mean, logvar], retain_graph=True)
              var_param_grads = torch.cat(var_param_grads, 1)
              var_params.backward(var_param_grads, retain_graph=True)
          else:
            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad],
                                                      t % args.print_every == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)        
      optimizer.step()      
      num_examples += batch_size          
      if t % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5
        print('Iters: %d, Epoch: %d, Batch: %d/%d, LR: %.4f, TrainARNLL: %.2f, TrainVAE_NLL: %.2f, TrainVAE_KL: %.4f, TrainVAE_NLLBnd: %.2f, TrainSVI_NLL: %.2f, TrainSVI_KL: %.4f, TrainSVI_NLLBnd: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.3f, Throughput: %.2f examples/sec' % 
              (t, epoch, b+1, len(train_loader), args.lr, train_nll_autoreg / num_examples, 
               train_nll_vae/num_examples, train_kl_vae / num_examples,
               (train_nll_vae + train_kl_vae)/num_examples,
               train_nll_svi/num_examples, train_kl_svi/ num_examples,
               (train_nll_svi + train_kl_svi)/num_examples,
               param_norm, best_val_nll, best_epoch, args.beta,
               num_examples / (time.time() - start_time)))
    print('--------------------------------')
    print('Checking validation perf...')
    val_nll = eval(val_loader, model, meta_optimizer)
    loss_stats.append(val_nll)
    if val_nll < best_val_nll:
      best_val_nll = val_nll
      best_epoch = epoch
      checkpoint = {
        'args': args.__dict__,
        'model': model,
        'optimizer': optimizer,
        'loss_stats': loss_stats
      }
      print('Saving checkpoint to %s' % args.checkpoint_path)
      torch.save(checkpoint, args.checkpoint_path)
        
    
def eval(data, model, meta_optimizer):
  model.eval()
  num_examples = 0
  total_nll_autoreg = 0.
  total_nll_vae = 0.
  total_kl_vae = 0.
  total_nll_svi = 0.
  total_kl_svi = 0.
  for datum in data:
    img, _ = datum
    batch_size = img.size(0)
    img = Variable(img.cuda())
    if args.model == 'autoreg':
      preds = model._dec_forward(img, None)
      nll_autoreg = utils.log_bernoulli_loss(preds, img)
      total_nll_autoreg += nll_autoreg.data[0]*batch_size
    elif args.model == 'svi':
      mean_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
      logvar_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
      var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
      mean_svi_final, logvar_svi_final = var_params_svi
      z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
      preds = model._dec_forward(img, z_samples)
      nll_svi = utils.log_bernoulli_loss(preds, img)
      total_nll_svi += nll_svi.data[0]*batch_size
      kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
      total_kl_svi += kl_svi.data[0]*batch_size      
    else:      
      mean, logvar = model._enc_forward(img)
      z_samples = model._reparameterize(mean, logvar)
      preds = model._dec_forward(img, z_samples)      
      nll_vae = utils.log_bernoulli_loss(preds, img)
      total_nll_vae += nll_vae.data[0]*batch_size
      kl_vae = utils.kl_loss_diag(mean, logvar)
      total_kl_vae += kl_vae.data[0]*batch_size
      if args.model == 'savi':
        mean_svi = Variable(mean.data, requires_grad = True)
        logvar_svi = Variable(logvar.data, requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)      
        preds = model._dec_forward(img, z_samples.detach())
        nll_svi = utils.log_bernoulli_loss(preds, img)
        total_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        total_kl_svi += kl_svi.data[0]*batch_size
        mean, logvar = mean_svi_final, logvar_svi_final
    num_examples += batch_size

  nll_autoreg = total_nll_autoreg / num_examples
  nll_vae = total_nll_vae/ num_examples
  kl_vae = total_kl_vae / num_examples
  nll_bound_vae = (total_nll_vae + total_kl_vae)/num_examples
  nll_svi = total_nll_svi/num_examples
  kl_svi = total_kl_svi/num_examples
  nll_bound_svi = (total_nll_svi + total_kl_svi)/num_examples
  print('AR NLL: %.4f, VAE NLL: %.4f, VAE KL: %.4f, VAE NLL BOUND: %.4f, SVI PPL: %.4f, SVI KL: %.4f, SVI NLL BOUND: %.4f' %
        (nll_autoreg, nll_vae, kl_vae, nll_bound_vae, nll_svi, kl_svi, nll_bound_svi))
  model.train()
  if args.model == 'autoreg':
    return nll_autoreg
  elif args.model == 'vae':
    return nll_bound_vae
  elif args.model == 'savi' or args.model == 'svi':
    return nll_bound_svi


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
