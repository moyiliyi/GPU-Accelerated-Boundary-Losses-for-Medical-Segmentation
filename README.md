# GPU-Accelerated-Boundary-Losses-for-Medical-Segmentation

## Introduction

In medical image segmentation tasks, many works utilized **Distance Transform Maps** to facilitate labeling around boundaries.  [https://github.com/JunMa11/SegWithDistMap] provided a collection of DTM-based losses. However, in these implementations, the euclidean distance transform is calculated from `scipy.ndimage.distance_transform_edt`. This is extremely slow and heavily relies on CPU resources. The training time for an epoch x10 in my experiments and the GPU utilization was ~ 0% for most of the time.

Here we provided a GPU accelerated implementation of Hausdorff loss and SDF loss based on the kornia library [2]. It uses convolutional to approximate the euclidean distance transform [3].

Usage examples:
```python
from losses import HDLoss
hd_loss = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=False)
loss = hd_loss(input, target)
```

Another interesting paper is here [4]. It provided a CUDA based euclidean distance transform implementation. The global and shared memory are carefully designed for Meijster Algorithm.


## Reference
[1] Ma, Jun, et al. "How distance transform maps boost segmentation CNNs: an empirical study." Medical Imaging with Deep Learning. PMLR, 2020.
[2] Riba, Edgar, et al. "Kornia: an open source differentiable computer vision library for pytorch." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020.
[3] Duc Duy Pham, Gurbandurdy Dovletov, and Josef Pauli. A differentiable convolutional distance transform layer for improved image segmentation. Pattern Recognition, 12544:432 – 444, 2020.
[4] de Assis Zampirolli, Francisco, and Leonardo Filipe. "A fast CUDA-based implementation for the Euclidean distance transform." 2017 International Conference on High Performance Computing & Simulation (HPCS). IEEE, 2017.