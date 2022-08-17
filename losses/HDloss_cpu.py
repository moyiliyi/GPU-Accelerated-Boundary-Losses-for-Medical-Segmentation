import torch
import numpy as np
from typing import Callable, List, Optional, Sequence, Union
from torch.nn.modules.loss import _Loss
from scipy.ndimage import distance_transform_edt as distance
from monai.utils import LossReduction
from monai.networks import one_hot
from joblib import delayed, Parallel

def compute_dtm01(img_gt, out_shape, device=None):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """
    
    normalized_dtm = np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        # ignore background ops choosed before
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)
    normalized_dtm = torch.Tensor(normalized_dtm).float().cuda(device)
    
    #from copy import deepcopy
    #normalized_dtm_npy = deepcopy(normalized_dtm)
    #normal_fuc = lambda x: x/np.max(x)
    #normalized_dtm = np.array([[normal_fuc( distance(posmask.astype(np.bool)) ) \
    #                             if posmask.astype(np.bool).any() \
    #                             else np.zeros(posmask.shape) \
    #                            for posmask in img_gt_b ] \
    #                            for img_gt_b in img_gt])
    """
    def _gen_one_map(ind):
        posmask = img_gt[int(ind/out_shape[1])][ind%out_shape[1]]
        posmask = posmask.astype(np.bool)
        if posmask.any():
            posdis = distance(posmask)
            out = posdis/np.max(posdis)
        else:
            out = np.zeros(posmask.shape)
        return out
    
    normalized_dtm =  Parallel(n_jobs=30, backend='threading')(delayed(_gen_one_map)(x) for x in range(out_shape[0]*out_shape[1]))
    normalized_dtm = torch.Tensor(normalized_dtm).float().cuda(device)
    normalized_dtm = normalized_dtm.view(out_shape)
    """
    #assert(normalized_dtm_npy == normalized_dtm).all()
    return normalized_dtm

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        #ignore background ops choosed before
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft - gt.float()) ** 2
    s_dtm = seg_dtm ** 2
    g_dtm = gt_dtm ** 2
    dtm = s_dtm + g_dtm
    if len(delta_s.shape) == 5: # B,C,H,W,D
        multipled = torch.einsum('bcxyz, bcxyz->bcxyz', delta_s, dtm)
    elif len(delta_s.shape) == 4: # B,C,H,W
        multipled = torch.einsum('bcxy, bcxy->bcxy', delta_s, dtm)
    else:
        raise RuntimeError("Got Error dim in HD Loss {}".format(delta_s.shape))
    #multipled = multipled.mean()

    return multipled
    

    
class HDLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

               # - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

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

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        with torch.no_grad():
            # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
            gt_dtm = compute_dtm01(target.cpu().numpy(), input.shape, input.device.index)
            #gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(input.device.index)
            seg_dtm = compute_dtm01(input.cpu().numpy()>0.5, input.shape, input.device.index)
            #seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(input.device.index)
        loss_hd = hd_loss(input, target, seg_dtm, gt_dtm)
        if self.reduction == LossReduction.MEAN.value:
            loss_hd = torch.mean(loss_hd)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            loss_hd = torch.sum(loss_hd)  # sum over the batch and channel dims
        #elif self.reduction == LossReduction.NONE.value:
        #    # If we are not computing voxelwise loss components at least
        #    # make sure a none reduction maintains a broadcastable shape
        #    broadcast_shape = list(loss_hd.shape[0:2]) + [1] * (len(input.shape) - 2)
        #    loss_hd = loss_hd.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return loss_hd