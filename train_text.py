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
from optim_n2n import OptimN2N
from data import Dataset
from models_text import RNNVAE
import utils

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='data/yahoo/yahoo-train.hdf5')
parser.add_argument('--val_file', default='data/yahoo/yahoo-val.hdf5')
parser.add_argument('--test_file', default='data/yahoo/yahoo-test.hdf5')
parser.add_argument('--train_from', default='')

# Model options
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_word_dim', default=512, type=int)
parser.add_argument('--enc_h_dim', default=1024, type=int)
parser.add_argument('--enc_num_layers', default=1, type=int)
parser.add_argument('--dec_word_dim', default=512, type=int)
parser.add_argument('--dec_h_dim', default=1024, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.5, type=float)
parser.add_argument('--model', default='savae', type=str, choices = ['vae', 'autoreg', 'savae', 'svi'])
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--train_kl', default=1, type=int)

# Optimization options
parser.add_argument('--checkpoint_path', default='baseline.pt')
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--min_epochs', default=15, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--svi_steps', default=20, type=int)
parser.add_argument('--svi_lr1', default=1, type=float)
parser.add_argument('--svi_lr2', default=1, type=float)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--decay', default=0, type=int)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--lr', default=1, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--svi_max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=2, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--test', type=int, default=0)

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)
  print('Train data: %d batches' % len(train_data))
  print('Val data: %d batches' % len(val_data))
  print('Word vocab size: %d' % vocab_size)
  if args.slurm == 0:
    cuda.set_device(args.gpu)
  if args.train_from == '':
    model = RNNVAE(vocab_size = vocab_size,
                   enc_word_dim = args.enc_word_dim,
                   enc_h_dim = args.enc_h_dim,
                   enc_num_layers = args.enc_num_layers,
                   dec_word_dim = args.dec_word_dim,
                   dec_h_dim = args.dec_h_dim,
                   dec_num_layers = args.dec_num_layers,
                   dec_dropout = args.dec_dropout,
                   latent_dim = args.latent_dim,
                   mode = args.model)
    for param in model.parameters():    
      param.data.uniform_(-0.1, 0.1)      
  else:
    print('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
      
  print("model architecture")
  print(model)

  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

  if args.warmup == 0:
    args.beta = 1.
  else:
    args.beta = 0.1
    
  criterion = nn.NLLLoss()
  model.cuda()
  criterion.cuda()
  model.train()

  def variational_loss(input, sents, model, z = None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)
    preds = model._dec_forward(sents, z_samples)
    nll = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(preds.size(1))])
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + args.beta*kl

  update_params = list(model.dec.parameters())
  meta_optimizer = OptimN2N(variational_loss, model, update_params, eps = args.eps, 
                            lr = [args.svi_lr1, args.svi_lr2],
                            iters = args.svi_steps, momentum = args.momentum,
                            acc_param_grads= args.train_n2n == 1,  
                            max_grad_norm = args.svi_max_grad_norm)
  if args.test == 1:
    args.beta = 1
    test_data = Dataset(args.test_file)    
    eval(test_data, model, meta_optimizer)
    exit()
    
  t = 0
  best_val_nll = 1e5
  best_epoch = 0
  val_stats = []
  epoch = 0
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll_vae = 0.
    train_nll_autoreg = 0.
    train_kl_vae = 0.
    train_nll_svi = 0.
    train_kl_svi = 0.
    train_kl_init_final = 0.
    num_sents = 0
    num_words = 0
    b = 0
    
    for i in np.random.permutation(len(train_data)):
      if args.warmup > 0:
        args.beta = min(1, args.beta + 1./(args.warmup*len(train_data)))
      
      sents, length, batch_size = train_data[i]
      if args.gpu >= 0:
        sents = sents.cuda()
      b += 1
      
      optimizer.zero_grad()
      if args.model == 'autoreg':
        preds = model._dec_forward(sents, None, True)
        nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_autoreg += nll_autoreg.data[0]*batch_size
        nll_autoreg.backward()
      elif args.model == 'svi':        
        mean_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        logvar_svi = Variable(0.1*torch.zeros(batch_size, args.latent_dim).cuda(), requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents,
                                                  b % args.print_every == 0)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
        preds = model._dec_forward(sents, z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        train_kl_svi += kl_svi.data[0]*batch_size      
        var_loss = nll_svi + args.beta*kl_svi          
        var_loss.backward(retain_graph = True)
      else:
        mean, logvar = model._enc_forward(sents)
        z_samples = model._reparameterize(mean, logvar)
        preds = model._dec_forward(sents, z_samples)
        nll_vae = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_vae += nll_vae.data[0]*batch_size
        kl_vae = utils.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.data[0]*batch_size        
        if args.model == 'vae':
          vae_loss = nll_vae + args.beta*kl_vae          
          vae_loss.backward(retain_graph = True)
        if args.model == 'savae':
          var_params = torch.cat([mean, logvar], 1)        
          mean_svi = Variable(mean.data, requires_grad = True)
          logvar_svi = Variable(logvar.data, requires_grad = True)
          var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents,
                                                  b % args.print_every == 0)
          mean_svi_final, logvar_svi_final = var_params_svi
          z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
          preds = model._dec_forward(sents, z_samples)
          nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
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
              train_kl_init_final += kl_init_final.data[0]*batch_size
              kl_init_final.backward(retain_graph = True)              
            else:
              vae_loss = nll_vae + args.beta*kl_vae
              var_param_grads = torch.autograd.grad(vae_loss, [mean, logvar], retain_graph=True)
              var_param_grads = torch.cat(var_param_grads, 1)
              var_params.backward(var_param_grads, retain_graph=True)              
          else:
            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad],
                                                      b % args.print_every == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)        
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * length
      
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5
        print('Iters: %d, Epoch: %d, Batch: %d/%d, LR: %.4f, TrainARPPL: %.2f, TrainVAE_PPL: %.2f, TrainVAE_KL: %.4f, TrainVAE_PPLBnd: %.2f, TrainSVI_PPL: %.2f, TrainSVI_KL: %.4f, TrainSVI_PPLBnd: %.2f, KLInitFinal: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.4f, Throughput: %.2f examples/sec' % 
              (t, epoch, b+1, len(train_data), args.lr, np.exp(train_nll_autoreg / num_words), 
               np.exp(train_nll_vae/num_words), train_kl_vae / num_sents,
               np.exp((train_nll_vae + train_kl_vae)/num_words),
               np.exp(train_nll_svi/num_words), train_kl_svi/ num_sents,
               np.exp((train_nll_svi + train_kl_svi)/num_words), train_kl_init_final / num_sents,
               param_norm, best_val_nll, best_epoch, args.beta,
               num_sents / (time.time() - start_time)))
          
    print('--------------------------------')
    print('Checking validation perf...')
    val_nll = eval(val_data, model, meta_optimizer)
    val_stats.append(val_nll)
    if val_nll < best_val_nll:
      best_val_nll = val_nll
      best_epoch = epoch
      model.cpu()
      checkpoint = {
        'args': args.__dict__,
        'model': model,
        'val_stats': val_stats
      }
      print('Savaeng checkpoint to %s' % args.checkpoint_path)      
      torch.save(checkpoint, args.checkpoint_path)
      model.cuda()
    else:
      if epoch >= args.min_epochs:
        args.decay = 1
    if args.decay == 1:
      args.lr = args.lr*0.5      
      for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
      if args.lr < 0.03:
        break
      
def eval(data, model, meta_optimizer):
    
  model.eval()
  criterion = nn.NLLLoss().cuda() 
  num_sents = 0
  num_words = 0
  total_nll_autoreg = 0.
  total_nll_vae = 0.
  total_kl_vae = 0.
  total_nll_svi = 0.
  total_kl_svi = 0.
  best_svi_loss = 0.
  for i in range(len(data)):
    sents, length, batch_size = data[i]
    num_words += batch_size*length
    num_sents += batch_size
    if args.gpu >= 0:
      sents = sents.cuda()
    if args.model == 'autoreg':
      preds = model._dec_forward(sents, None, True)
      nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_autoreg += nll_autoreg.data[0]*batch_size
    elif args.model == 'svi':
      mean_svi = Variable(0.1*torch.randn(batch_size, args.latent_dim).cuda(), requires_grad = True)
      logvar_svi = Variable(0.1*torch.randn(batch_size, args.latent_dim).cuda(), requires_grad = True)
      var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
      mean_svi_final, logvar_svi_final = var_params_svi
      z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
      preds = model._dec_forward(sents, z_samples)
      nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_svi += nll_svi.data[0]*batch_size
      kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
      total_kl_svi += kl_svi.data[0]*batch_size
      mean, logvar = mean_svi_final, logvar_svi_final
    else:
      mean, logvar = model._enc_forward(sents)
      z_samples = model._reparameterize(mean, logvar)
      preds = model._dec_forward(sents, z_samples)
      nll_vae = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_vae += nll_vae.data[0]*batch_size
      kl_vae = utils.kl_loss_diag(mean, logvar)
      total_kl_vae += kl_vae.data[0]*batch_size        
      if args.model == 'savae':
        mean_svi = Variable(mean.data, requires_grad = True)
        logvar_svi = Variable(logvar.data, requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
        preds = model._dec_forward(sents, z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        total_nll_svi += nll_svi.data[0]*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        total_kl_svi += kl_svi.data[0]*batch_size      
        mean, logvar = mean_svi_final, logvar_svi_final
  ppl_autoreg = np.exp(total_nll_autoreg / num_words)
  ppl_vae = np.exp(total_nll_vae/ num_words)
  kl_vae = total_kl_vae / num_sents
  ppl_bound_vae = np.exp((total_nll_vae + total_kl_vae)/num_words)
  ppl_svi = np.exp(total_nll_svi/num_words)
  kl_svi = total_kl_svi/num_sents
  ppl_bound_svi = np.exp((total_nll_svi + total_kl_svi)/num_words)
  print('AR PPL: %.4f, VAE PPL: %.4f, VAE KL: %.4f, VAE PPL BOUND: %.4f, SVI PPL: %.4f, SVI KL: %.4f, SVI PPL BOUND: %.4f' %
        (ppl_autoreg, ppl_vae, kl_vae, ppl_bound_vae, ppl_svi, kl_svi, ppl_bound_svi))
  model.train()
  if args.model == 'autoreg':
    return ppl_autoreg
  elif args.model == 'vae':
    return ppl_bound_vae
  elif args.model == 'savae' or args.model == 'svi':
    return ppl_bound_svi

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
