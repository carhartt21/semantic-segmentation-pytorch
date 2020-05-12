import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import load_url, label_to_onehot

batch_norm_2d = torch.nn.BatchNorm2d
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class ModuleHelper:

    @staticmethod
    def bn_relu(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            batch_norm_2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def batch_norm_2d(*args, **kwargs):
        return batch_norm_2d


class SpatialGather(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1)  # batch x h x w x c 
            gt_probs = F.normalize(gt_probs, p=1, dim=2)  # batch x k x hw
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context
        else:
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            # batch x h x w x c
            feats = feats.permute(0, 2, 1)
            # batch x k x hw
            probs = F.softmax(self.scale * probs, dim=2)
            # batch x k x c
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
            return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
        use_gt            : whether use the ground truth label map to compute the similarity map
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None, use_gt=False, align_corners=False):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners
        self.use_gt = use_gt
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        # function ψ
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.key_channels, bn_type=bn_type),
        )
        # function ϕ
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.key_channels, bn_type=bn_type),
        )
        # function δ
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.key_channels, bn_type=bn_type),
        )
        # function ρ
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.bn_relu(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)  # ψ(X)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)  # ϕ(fk)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)  # δ(fk)
        value = value.permute(0, 2, 1)
        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(
                1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(
                0, 2, 3, 1).view(batch_size, h*w, -1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)  # similarity matrix W
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)  # channel softmax normalization
        context = torch.matmul(sim_map, value)  # contextual representation X^o
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)  # ρ
        # print(self.f_up[1].weight.squeeze())
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None,
                 use_gt=False,
                 align_corners=False):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type,
                                                     use_gt=use_gt,
                                                     align_corners=align_corners)


class SpatialOCR(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    use_gt: whether use the ground-truth label to compute the ideal object contextual representations.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None,
                 use_gt=False,
                 align_corners=False):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type=bn_type,
                                                           use_gt=use_gt,
                                                           align_corners=align_corners)
        _in_channels = 2 * in_channels

        # function g
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.bn_relu(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)  # randomly zeros out entire channels with probability "dropout"
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        # compute the object contextual representation X^o
        context = self.object_context_block(feats, proxy_feats, gt_label)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))  # concatenate X^o and X and apply g

        return output
