"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from common_net import *


class COCOSharedDis(nn.Module):
  def __init__(self, params):
    super(COCOSharedDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_front_layer = params['n_front_layer']
    n_shared_layer = params['n_shared_layer']
    self.model_A, tch = self._make_front_net(ch, input_dim_a, n_front_layer, n_shared_layer==0)
    self.model_B, tch = self._make_front_net(ch, input_dim_b, n_front_layer, n_shared_layer==0)
    self.model_S = self._make_shared_net(tch, n_shared_layer)

  def _make_front_net(self, ch, input_dim, n_layer, add_classifier_layer=False):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=7, stride=2, padding=3)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    if add_classifier_layer:
      model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model), tch

  def _make_shared_net(self, ch, n_layer):
    model = []
    if n_layer == 0:
      return nn.Sequential(*model)
    tch = ch
    for i in range(0, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)
    self.model_S.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_S(self.model_A(x_A))
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    out_B = self.model_S(self.model_B(x_B))
    out_B = out_B.view(-1)
    outs_B = []
    outs_B.append(out_B)
    return outs_A, outs_B

class COCOCatRegDis(nn.Module):
  def __init__(self, params):
    super(COCOCatRegDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    self.number_of_categories_a = params['categorise_a']
    self.number_of_categories_b = params['categorise_b']
    n_layer = params['n_layer']
    last_kernel_size = params['last_kernel_size']
    self.model_A, self.cat_A, self.reg_A = self._make_net(ch, input_dim_a, n_layer, self.number_of_categories_a, last_kernel_size)
    self.model_B, self.cat_B, self.reg_B = self._make_net(ch, input_dim_b, n_layer, self.number_of_categories_b, last_kernel_size)

  def _make_net(self, ch, input_dim, n_layer, number_of_categories, last_kernel_size):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    last_cat_layer = nn.Conv2d(tch, number_of_categories, kernel_size=last_kernel_size, stride=1, padding=0)  # 1
    last_reg_layer = nn.Conv2d(tch, 1, kernel_size=last_kernel_size, stride=1, padding=0)  # 1
    return nn.Sequential(*model), nn.Sequential(last_cat_layer), nn.Sequential(last_reg_layer)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)
    self.cat_A.cuda(gpu)
    self.cat_B.cuda(gpu)
    self.reg_A.cuda(gpu)
    self.reg_B.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_A(x_A)
    out_cat_A = self.cat_A(out_A)
    out_cat_A = out_cat_A.transpose(1,3)
    out_cat_A = out_cat_A.transpose(1,2).contiguous()
    out_cat_A = out_cat_A.view(-1, self.number_of_categories_a)
    outs_cat_A = []
    outs_cat_A.append(out_cat_A)
    out_reg_A = self.reg_A(out_A)
    out_reg_A = out_reg_A.transpose(1,3)
    out_reg_A = out_reg_A.transpose(1,2).contiguous()
    out_reg_A = out_reg_A.view(-1, 1)
    outs_reg_A = []
    outs_reg_A.append(out_reg_A)

    out_B = self.model_B(x_B)
    out_cat_B = self.cat_B(out_B)
    out_cat_B = out_cat_B.transpose(1,3)
    out_cat_B = out_cat_B.transpose(1,2).contiguous()
    out_cat_B = out_cat_B.view(-1, self.number_of_categories_b)
    outs_cat_B = []
    outs_cat_B.append(out_cat_B)
    out_reg_B = self.reg_B(out_B)
    out_reg_B = out_reg_B.transpose(1,3)
    out_reg_B = out_reg_B.transpose(1,2).contiguous()
    out_reg_B = out_reg_B.view(-1, 1)
    outs_reg_B = []
    outs_reg_B.append(out_reg_B)

    return outs_cat_A, outs_reg_A, outs_cat_B, outs_reg_B

class COCOCategorialDis(nn.Module):
  def __init__(self, params):
    super(COCOCategorialDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    self.number_of_categories_a = params['categorise_a']
    self.number_of_categories_b = params['categorise_b']
    n_layer = params['n_layer']
    last_kernel_size = params['last_kernel_size']
    self.model_A = self._make_net(ch, input_dim_a, n_layer, self.number_of_categories_a, last_kernel_size)
    self.model_B = self._make_net(ch, input_dim_b, n_layer, self.number_of_categories_b, last_kernel_size)

  def _make_net(self, ch, input_dim, n_layer, number_of_categories, last_kernel_size):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, number_of_categories, kernel_size=last_kernel_size, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_A(x_A)
    out_A = out_A.transpose(1,3)
    out_A = out_A.transpose(1,2).contiguous()
    out_A = out_A.view(-1, self.number_of_categories_a)
    outs_A = []
    outs_A.append(out_A)
    out_B = self.model_B(x_B)
    out_B = out_B.transpose(1,3)
    out_B = out_B.transpose(1,2).contiguous()
    out_B = out_B.view(-1, self.number_of_categories_b)
    outs_B = []
    outs_B.append(out_B)
    return outs_A, outs_B

class COCODis(nn.Module):
  def __init__(self, params):
    super(COCODis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_layer = params['n_layer']
    self.model_A = self._make_net(ch, input_dim_a, n_layer)
    self.model_B = self._make_net(ch, input_dim_b, n_layer)

  def _make_net(self, ch, input_dim, n_layer):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_A(x_A)
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    out_B = self.model_B(x_B)
    out_B = out_B.view(-1)
    outs_B = []
    outs_B.append(out_B)
    return outs_A, outs_B

class COCOResGen(nn.Module):
  def __init__(self, params):
    super(COCOResGen, self).__init__()
    output_dim_a = params['input_dim_a']
    output_dim_b = params['input_dim_b']
    if 1 == params['use_xy']:
        input_dim_a = 2 + output_dim_a
        input_dim_b = 2 + output_dim_b
    else:
        input_dim_a = output_dim_a
        input_dim_b = output_dim_b

    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    encA = []
    encB = []
    # Encoders
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch)]
      encB += [INSResBlock(tch, tch)]
    # Shared
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch)]
    enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch)]
    # Decoders
    decA = []
    decB = []
    for i in range(0, n_gen_res_blk):
      decA += [INSResBlock(tch, tch)]
      decB += [INSResBlock(tch, tch)]
    for i in range(0, n_gen_front_blk-1):
      decA += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared

  def cuda(self, gpu):
    self.encode_A.cuda(gpu)
    self.encode_B.cuda(gpu)
    self.enc_shared.cuda(gpu)
    self.dec_shared.cuda(gpu)
    self.decode_A.cuda(gpu)
    self.decode_B.cuda(gpu)

  def share_memory(self):
    self.encode_A.share_memory()
    self.encode_B.share_memory()
    self.enc_shared.share_memory()
    self.dec_shared.share_memory()
    self.decode_A.share_memory()
    self.decode_B.share_memory()

class COCOResGen2(nn.Module):
  def __init__(self, params):
    super(COCOResGen2, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    encA = []
    encB = []
    # Encoders
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch)]
      encB += [INSResBlock(tch, tch)]
    # Shared
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch)]
    enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch)]
    # Decoders
    decA = []
    decB = []
    for i in range(0, n_gen_res_blk):
      decA += [INSResBlock(tch, tch)]
      decB += [INSResBlock(tch, tch)]
    for i in range(0, n_gen_front_blk-1):
      decA += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared

  def cuda(self, gpu):
    self.encode_A.cuda(gpu)
    self.encode_B.cuda(gpu)
    self.enc_shared.cuda(gpu)
    self.dec_shared.cuda(gpu)
    self.decode_A.cuda(gpu)
    self.decode_B.cuda(gpu)

  def share_memory(self):
    self.encode_A.share_memory()
    self.encode_B.share_memory()
    self.enc_shared.share_memory()
    self.dec_shared.share_memory()
    self.decode_A.share_memory()
    self.decode_B.share_memory()

