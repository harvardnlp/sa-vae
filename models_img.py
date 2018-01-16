#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def he_init(m):
  s = np.sqrt(2./ m.in_features)
  m.weight.data.normal_(0, s)

class GatedMaskedConv2d(nn.Module):
  def __init__(self, in_dim, out_dim=None, kernel_size = 3, mask = 'B'):
    super(GatedMaskedConv2d, self).__init__()
    if out_dim is None:
      out_dim = in_dim    
    self.dim = out_dim
    self.size = kernel_size
    self.mask = mask
    pad = self.size // 2
    
    #vertical stack    
    self.v_conv = nn.Conv2d(in_dim, 2*self.dim, kernel_size=(pad+1, self.size))
    self.v_pad1 = nn.ConstantPad2d((pad, pad, pad, 0), 0)
    self.v_pad2 = nn.ConstantPad2d((0, 0, 1, 0), 0)
    self.vh_conv = nn.Conv2d(2*self.dim, 2*self.dim, kernel_size = 1)

    #horizontal stack
    self.h_conv = nn.Conv2d(in_dim, 2*self.dim, kernel_size=(1, pad+1))
    self.h_pad1 = nn.ConstantPad2d((self.size // 2, 0, 0, 0), 0)
    self.h_pad2 = nn.ConstantPad2d((1, 0, 0, 0), 0)
    self.h_conv_res = nn.Conv2d(self.dim, self.dim, 1)

  def forward(self, v_map, h_map):
    v_out = self.v_pad2(self.v_conv(self.v_pad1(v_map)))[:, :, :-1, :]
    v_map_out = F.tanh(v_out[:, :self.dim])*F.sigmoid(v_out[:, self.dim:])
    vh = self.vh_conv(v_out)
    
    h_out = self.h_conv(self.h_pad1(h_map))
    if self.mask == 'A':
      h_out = self.h_pad2(h_out)[:, :, :, :-1]
    h_out = h_out + vh    
    h_out = F.tanh(h_out[:, :self.dim])*F.sigmoid(h_out[:, self.dim:])
    h_map_out = self.h_conv_res(h_out)
    if self.mask == 'B':
      h_map_out = h_map_out + h_map
    return v_map_out, h_map_out

class StackedGatedMaskedConv2d(nn.Module):
  def __init__(self, 
               img_size = [1, 28, 28], layers = [64,64,64],
               kernel_size = [7,7,7], latent_dim=64, latent_feature_map = 1):
    super(StackedGatedMaskedConv2d, self).__init__()
    input_dim = img_size[0]
    self.conv_layers = []
    if latent_feature_map > 0:
      self.latent_feature_map = latent_feature_map
      self.z_linear = nn.Linear(latent_dim, latent_feature_map*28*28)    
    for i in range(len(kernel_size)):
      if i == 0:
        self.conv_layers.append(GatedMaskedConv2d(input_dim+latent_feature_map,
                                                  layers[i],  kernel_size[i], 'A'))
      else:
        self.conv_layers.append(GatedMaskedConv2d(layers[i-1], layers[i],  kernel_size[i]))
        
    self.modules = nn.ModuleList(self.conv_layers)
      
  def forward(self, img, q_z=None):
    if q_z is not None:
      z_img = self.z_linear(q_z) 
      z_img = z_img.view(img.size(0), self.latent_feature_map, img.size(2), img.size(3))
    
    for i in range(len(self.conv_layers)):
      if i == 0:
        if q_z is not None:
          v_map = torch.cat([img, z_img], 1)
        else:
          v_map = img
        h_map = v_map
      v_map, h_map = self.conv_layers[i](v_map, h_map)
    return h_map
  
class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True, mask=None,
               kernel_size = 3, padding = 1):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()
    if mask is None:
      self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
      self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding)
    else:
      self.conv1 = MaskedConv2d(mask, in_dim, out_dim, kernel_size=kernel_size, padding=padding)
      self.conv2 = MaskedConv2d(mask, out_dim, out_dim, kernel_size=kernel_size, padding=padding)
    self.with_batchnorm = with_batchnorm
    if with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim)
      self.bn2 = nn.BatchNorm2d(out_dim)
    self.with_residual = with_residual
    if in_dim == out_dim or not with_residual:
      self.proj = None
    else:
      self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

  def forward(self, x):
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out

class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_center=False, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (include_center == True):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask.cuda()
        return super(MaskedConv2d, self).forward(x)
     
class CNNVAE(nn.Module):
  def __init__(self,
               img_size = [1,28,28],
               latent_dim = 32,
               enc_layers = [64,64,64],
               dec_kernel_size = [7,7,7],
               dec_layers= [64,64,64],
               latent_feature_map = 4):
    super(CNNVAE, self).__init__()
    enc_modules = []
    img_h = img_size[1]
    img_w = img_size[2]    
    for i in range(len(enc_layers)):
      if i == 0:
        input_dim = img_size[0]
      else:
        input_dim = enc_layers[i-1]
      enc_modules.append(ResidualBlock(input_dim, enc_layers[i]))
      enc_modules.append(nn.Conv2d(enc_layers[i], enc_layers[i], kernel_size=2, stride=2))
      
      img_h //= 2
      img_w //= 2
    latent_in_dim = img_h*img_w*enc_layers[-1]
    self.enc_cnn = nn.Sequential(*enc_modules)
    self.latent_linear_mean = nn.Linear(latent_in_dim, latent_dim)
    self.latent_linear_logvar = nn.Linear(latent_in_dim, latent_dim)   
    self.enc = nn.ModuleList([self.enc_cnn, self.latent_linear_mean, self.latent_linear_logvar])
    self.dec_cnn = StackedGatedMaskedConv2d(img_size=img_size, layers = dec_layers,
                                            latent_dim= latent_dim, kernel_size = dec_kernel_size,
                                            latent_feature_map = latent_feature_map)
    self.dec_linear = nn.Conv2d(dec_layers[-1], img_size[0], kernel_size = 1)
    self.dec = nn.ModuleList([self.dec_cnn, self.dec_linear])
    for m in self.modules():
      if isinstance(m, nn.Linear):
        he_init(m)
      
  def _enc_forward(self, img):
    img_code = self.enc_cnn(img)
    img_code = img_code.view(img.size(0), -1)
    self.img_code = img_code
    mean = self.latent_linear_mean(img_code)
    logvar = self.latent_linear_logvar(img_code)
    return mean, logvar
  
  def _reparameterize(self, mean, logvar, z = None):
    self.std = logvar.mul(0.5).exp()    
    if z is None:
      self.z = Variable(torch.FloatTensor(self.std.size()).normal_(0, 1).type_as(mean.data))
    else:
      self.z = z
    self.q_z = self.z*self.std + mean
    return self.q_z

  def _dec_forward(self, img, q_z):
    dec_cnn_output = self.dec_cnn(img, q_z)
    pred = F.sigmoid(self.dec_linear(dec_cnn_output))
    return pred
                   
