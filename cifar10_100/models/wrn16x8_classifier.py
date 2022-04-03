import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from utils.OT import SinkhornDistance
import numpy as np
from models.WideResnet import Wide_ResNet
import random


def mixup_process(out, y, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    y_a, y_b = y, y[indices]
    return out, y_a, y_b




def mixup_aligned(out, y, lam):
    # out shape = batch_size x 512 x 4 x 4 (cifar10/100)

    indices = np.random.permutation(out.size(0))
    feat1 = out.view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16
    feat2 = out[indices].view(out.shape[0], out.shape[1], -1) # batch_size x 512 x 16
    
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach()  # optimal plan batch x 16 x 16
    
    P = P*(out.size(2)*out.size(3)) # assignment matrix 

    align_mix = random.randint(0,1) # uniformly choose at random, which alignmix to perform
   
    if (align_mix == 0):
        # \tilde{A} = A'R^{T}
        f1 = torch.matmul(feat2, P.permute(0,2,1).cuda()).view(out.shape) 
        final = feat1.view(out.shape)*lam + f1*(1-lam)

    elif (align_mix == 1):
        # \tilde{A}' = AR
        f2 = torch.matmul(feat1, P.cuda()).view(out.shape).cuda()
        final = f2*lam + feat2.view(out.shape)*(1-lam)

    y_a, y_b = y,y[indices]

    return final, y_a, y_b




class WideResNet_classifier(nn.Module):
    def __init__(self, num_classes, z_dim=512):
        super(WideResNet_classifier, self).__init__()
        
        self.encoder = Wide_ResNet(depth=16,
                        widen_factor=8,
                        num_classes=10,
                        stride=1)
        
        self.classifier = nn.Linear(z_dim,num_classes)

        

    def forward(self, x, targets, lam, mode):
        
        if (mode == 'train'):
            
            layer_mix = random.randint(0,1)
            if layer_mix == 0:
                x,t_a,t_b = mixup_process(x, targets, lam)

            out = self.encoder(x)

            if layer_mix == 1:
                out,t_a,t_b = mixup_aligned(out, targets, lam)            

            out = F.avg_pool2d(out, 8)
            out = out.reshape(out.size(0), -1)
            cls_output = self.classifier(out)

            
            return cls_output, t_a, t_b

            
        
        elif (mode == 'test'):
            out = self.encoder(x)
            out = F.avg_pool2d(out, 8)
            out = out.reshape(out.size(0), -1)
            cls_output = self.classifier(out)

            return  cls_output
    
