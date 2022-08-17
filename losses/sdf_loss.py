import torch
from .korniadt import distance_transform, dilation, erosion
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
from monai.networks import one_hot

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from scipy import ndimage as ndi
from skimage import segmentation as skimage_seg

def compute_sdf_cpu(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    device = img_gt.device.index
    img_gt = img_gt.cpu().numpy()
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)+ 1e-5) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)+ 1e-5)
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                #assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                #assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return torch.from_numpy(normalized_sdf).float().cuda(device)

def find_boundaries_torch(mask):
    #if len(mask.shape) !=3:
    #    print('Warining finding mask shape:',len(mask.shape) ,'in find_boundaries_torch')
    background = 0
    footprint = torch.Tensor(ndi.generate_binary_structure(2, 1)).cuda(mask.device.index)
    boundaries = (dilation(mask.unsqueeze(0), torch.Tensor(3,3), footprint) != erosion(mask.unsqueeze(0), torch.Tensor(3,3), footprint))
    foreground_image = (mask != background)
    boundaries = torch.logical_and(boundaries, foreground_image)
    return boundaries[0]

def compute_sdf(img_gt, out_shape, kernel_size=5):
    #img_gt = img_gt.astype(np.uint8)
    #normalized_sdf = np.zeros(out_shape)
    device = img_gt.device.index
    normalized_sdf = torch.zeros(out_shape).cuda(device)
    if len(out_shape) == 5: # B,C,H,W,D
        for b in range(out_shape[0]): 
            posdis = distance_transform(1-img_gt[b].float(), kernel_size = kernel_size)# C,H,W,D
            negdis = distance_transform(img_gt[b].float(), kernel_size = kernel_size)
            posdis[~torch.isfinite(posdis)] = kernel_size
            negdis[~torch.isfinite(negdis)] = kernel_size
            
            #posdis_cpu = posdis.cpu().numpy()# C,H,W,D
            for c in range(out_shape[1]):
                boundary = find_boundaries_torch(posdis[c])
                #boundary2 = torch.from_numpy(skimage_seg.find_boundaries(posdis_cpu[c], mode='inner')).int().cuda(device)
                #print((boundary==boundary2).all())
                sdf = (negdis[c]-torch.min(negdis[c]))/(torch.max(negdis[c])-torch.min(negdis[c])+ 1e-5) \
                      - (posdis[c]-torch.min(posdis[c]))/(torch.max(posdis[c])-torch.min(posdis[c]+ 1e-5))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
    else:
        posdis = distance_transform(1-img_gt.float(), kernel_size = kernel_size)# C,H,W,D
        negdis = distance_transform(img_gt.float(), kernel_size = kernel_size)
        posdis[~torch.isfinite(posdis)] = kernel_size
        negdis[~torch.isfinite(negdis)] = kernel_size
        posdis_cpu = posdis.cpu().numpy()
        for c in range(out_shape[1]):
            boundary = torch.from_numpy(skimage_seg.find_boundaries(posdis_cpu[c], mode='inner')).int().cuda(device)
            sdf = (negdis[c]-np.min(negdis[c]))/(np.max(negdis[c])-np.min(negdis[c])+ 1e-5) \
                      - (posdis[c]-np.min(posdis[c]))/(np.max(posdis[c])-np.min(posdis[c])+ 1e-5)
            sdf[boundary==1] = 0
            normalized_sdf[c] = sdf
    return normalized_sdf

class SDF_loss(_Loss):
    def __init__(self,sigmoid=True, to_onehot_y=True, include_background=True):
        super(SDF_loss, self).__init__() 
        self.sigmoid = sigmoid
        self.to_onehot_y = to_onehot_y
        self.include_background = include_background
        
    def forward(self,input, input_dis, target):
        if self.sigmoid:
            input_dis = torch.tanh(input_dis)
        
        n_pred_ch = input.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                input_dis = input_dis[:, 1:]
                
        with torch.no_grad():
            gt_dis = compute_sdf(target, input_dis.shape)
            # gt_dis = torch.from_numpy(gt_dis).float().cuda()
        loss_dist = F.mse_loss(input_dis, gt_dis)
        return loss_dist
    