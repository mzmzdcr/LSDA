import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np

def binary_entropy(p):
    p = torch.clamp(p, min=1e-6, max=1-(1e-6))
    h = - p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
    return h

def multi_entropy(p):
    p = F.softmax(p, dim=1)
    h = - p * torch.log2(p)
    return h.sum()

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def dann_feat_loss(feat, label, ad_net, alpha, t):
    ad_out = ad_net(feat)
    return binary_adv_loss(ad_out, label, alpha, t, False)




def feature_adloss(input_list, dc_target, ad_net, device, random_layer=None, entropy=None, coeff=None, valuea=None, t=0.35):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy).to(device)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy).to(device)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * binary_adv_loss(ad_out, dc_target, valuea, t, True)) / torch.sum(weight).detach().item()
    else:
        return binary_adv_loss(ad_out, dc_target, valuea, t, False)

def feat_adout(input_list, ad_net, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    
    return ad_out

def src_loss(x, alpha, t):
    ent = binary_entropy(x) 
    aligned = torch.zeros_like(ent)
    notaligned = torch.ones_like(ent) * alpha[0]
    a = ent * 1.0
    a = torch.where(ent > t, aligned, notaligned)
    # a[x>0.5] = 0
    p = x * (1-a)/(1 - a * x)
    return p

def tar_loss(x, alpha, t):
    ent = binary_entropy(x) 
    aligned = torch.zeros_like(ent)
    notaligned = torch.ones_like(ent) * alpha[0]
    a = ent * 1.0
    a = torch.where(ent > t, aligned, notaligned)
    # a[x<0.5] = 0
    p = x/(x * a + 1 - a) 
    return p



def binary_adv_loss(x, y, alpha, t, flatten=False):
    bs = x.size(0) // 2
    src_p = src_loss(x[:bs], alpha, t)
    tar_p = tar_loss(x[bs:], alpha, t)
    p = torch.cat((src_p, tar_p), dim=0)

    if flatten:
        loss = nn.BCELoss(reduction='none')(p, y.unsqueeze(1).float())
    else:
        loss = nn.BCELoss()(p, y.unsqueeze(1).float())
    return loss


def CDAN_(input_list, ad_net, device, entropy=None, coeff=None, alpha=None, random_layer=None, t=0.35):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy).to(device)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy).to(device)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * binary_adv_loss(ad_out, dc_target, alpha, t, True)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)