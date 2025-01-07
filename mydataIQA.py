import tifffile as tiff
import glob
import torch.utils.data as data
import cv2
import torch
import os
import numpy as np
from tifffile import imread
from scipy.ndimage.interpolation import zoom
from csbdeep.utils import normalize


datamin, datamax = 0, 100  #


def np2Tensor(*args):
    def _np2Tensor(img):
        # print('np2Tensor img.shape', img.shape)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    
    return [_np2Tensor(a) for a in args]
 
# Inheritted from CARE
class PercentileNormalizer(object):
    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=torch.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs
    
    def before(self, img, axes):
        if len(axes) != img.ndim:
            raise ValueError
        channel = None if axes.find('C') == -1 else axes.find('C')
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img.detach().cpu().numpy(), self.pmin, axis=axes, keepdims=True).astype(np.float32, copy=False)
        self.ma = np.percentile(img.detach().cpu().numpy(), self.pmax, axis=axes, keepdims=True).astype(np.float32, copy=False)
        return (img - self.mi) / (self.ma - self.mi + 1e-20)
    
    def after(self, img):
        if not self.do_after():
            raise ValueError
        alpha = self.ma - self.mi
        beta = self.mi
        return (alpha * img + beta).astype(np.float32, copy=False)
    
    def do_after(self):
        return self._do_after
