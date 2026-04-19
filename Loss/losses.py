import torch
import math

def mse_loss(input, target, mask):
    diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0
    diff2[mask.flatten()] = 0.0
    result = torch.sum(diff2) / torch.sum(torch.logical_not(mask))
    return result

def mse_loss_each_batch(input, target, mask):
    diff2 = (input - target) ** 2.0
    diff2[mask] = 0.0
    result = torch.sum(torch.sum(diff2, dim=-1) / torch.sum(torch.logical_not(mask), dim=-1))
    return result

