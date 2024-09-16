import tifffile as tiff
# import nibabel as nib
# import random
import imageio
import glob
import torch.utils.data as data
import cv2
import torch
import os
import numpy as np
from tifffile import imread, imsave
from scipy.ndimage.interpolation import zoom
from csbdeep.utils import normalize, axes_dict, axes_check_and_normalize


datamin, datamax = 0, 100  #


def move_channel_for_backend(X, channel):
    return np.moveaxis(X, channel, -1)


def load_training_data(file, validation_split=0, axes=None, n_images=None, verbose=False):
    """Load training data from file in ``.npz`` format.

    The data file is expected to have the keys:

    - ``X``    : Array of training input images.
    - ``Y``    : Array of corresponding target images.
    - ``axes`` : Axes of the training images.


    Parameters
    ----------
    file : str
        File name
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    n_images : int, optional
        Can be used to limit the number of images loaded from data.
    verbose : bool, optional
        Can be used to display information about the loaded images.

    Returns
    -------
    tuple( tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`), (`X_val`, `Y_val`) of training and validation sets
        and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.

    """
    
    f = np.load(file)
    X, Y = f['X'], f['Y']
    print(Y.ndim, Y.shape)
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)
    
    assert X.ndim == Y.ndim
    assert len(axes) == X.ndim
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    
    if validation_split > 0:
        n_val = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:], Y[-n_val:]
        X, Y = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t, channel=channel)
        Y_t = move_channel_for_backend(Y_t, channel=channel)
    
    X = move_channel_for_backend(X, channel=channel)
    Y = move_channel_for_backend(Y, channel=channel)
    
    axes = axes.replace('C', '')  # remove channel
    
    axes = axes + 'C'
    # if backend_channels_last():
    #     axes = axes + 'C'
    # else:
    #     axes = axes[:1] + 'C' + axes[1:]
    
    data_val = (X_t, Y_t) if validation_split > 0 else None
    
    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split > 0 else 0
        image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
        n_dim = len(image_size)
        n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]
        
        print('number of training images:\t', n_train)
        print('number of validation images:\t', n_val)
        print('image size (%dD):\t\t' % n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)
    
    return (X, Y), data_val, axes


def np2Tensor(*args):
    def _np2Tensor(img):
        # print('np2Tensor img.shape', img.shape)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    
    return [_np2Tensor(a) for a in args]


def loadData(traindatapath, axes='SCYX', validation_split=0.05):
    print('Load data npz')
    if validation_split > 0:
        (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
    else:
        (X, Y), _, axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
        X_val, Y_val = 0, 1
    print(X.shape, Y.shape)
    return X, Y, X_val, Y_val


class SR(data.Dataset):
    def __init__(self, LRpath='', name='CCPs', 
                 rootdatapath='/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'):
        self.dir_demo = rootdatapath + 'test/%s/GT/' % name
        self.dir_demoLR = rootdatapath + 'test/%s/LR/' % name
        print('LRpath = ', LRpath)
        if LRpath == '' or LRpath == '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/GT/' % name:
            self.dir_input = self.dir_demo
            ext = '_GT.tif'
        else:
            self.dir_input = LRpath
        print('self.dir_input = ', self.dir_input)
        self.images_hr, self.images_sr, self.images_lr, self.name, self.filenames, \
            self.filenamesLR, self.filenamesinput = [], [], [], [], [], [], []
        
        if os.path.exists(self.dir_input + 'im1_noNorm_noUint8.tif'):
            ext = '_noNorm_noUint8.tif'
        elif os.path.exists(self.dir_input + 'im1_LR.tif'):
            ext = '_LR.tif'
        elif os.path.exists(self.dir_input + 'im1_LR.png'):
            ext = '_LR.png'
            
        for fi in range(1, 101):  # for fi in range(43, 44):  # 
            self.filenames.append(self.dir_demo + 'im%d_GT.tif' %(fi))
            self.filenamesinput.append(self.dir_input + 'im%d%s' %(fi, ext))
            self.filenamesLR.append(self.dir_demoLR + 'im%d_LR.tif' %(fi))
            self.name.append('im%d_GT%s' %(fi, ext))
            
    def __getitem__(self, idx):
        lrnm, srnm, hrnm, filename = self.filenamesLR[idx], self.filenamesinput[idx], self.filenames[idx], self.name[idx]

        lr = tiff.imread(lrnm)  # np.array(Image.open(lrnm))
        hr = tiff.imread(hrnm)  # np.array(Image.open(hrnm))
        if '.png' in srnm:
            sr = cv2.cvtColor(np.squeeze(cv2.imread(srnm)), cv2.COLOR_BGR2GRAY)
        else:
            sr = tiff.imread(srnm)  # np.array(Image.open(srnm))        
        
        # print('lr sr hr.shape = ', lr.shape, sr.shape, hr.shape)
        # print('\n', filename)
        if len(sr.shape) < 3:
            sr = np.expand_dims(sr, -1)
        if len(lr.shape) < 3:
            lr = np.expand_dims(lr, -1)
        if len(hr.shape) < 3:
            hr = np.expand_dims(hr, -1)
                
        hr = normalize(hr, datamin, datamax, clip=True) 
        sr = normalize(sr, datamin, datamax, clip=True) 
        lr = normalize(lr, datamin, datamax, clip=True) 
        # print(np.max(lr), ' = np.max(lr)')
        # print('lr.shape = (256, 256, 1) ', lr.shape)
        pair = (lr, sr, hr)
        pair_t = np2Tensor(*pair)
        # print('pair_t[0].shape = ', pair_t[0].shape, pair_t[1].shape)

        return pair_t[0], pair_t[1], pair_t[2], filename
    
    def getitem_IQA(self, idx):
        lrnm, srnm, hrnm, filename = self.filenamesLR[idx], self.filenamesinput[idx], self.filenames[idx], self.name[idx]

        lr = tiff.imread(lrnm)
        hr = tiff.imread(hrnm)
        if '.png' in srnm:
            sr = cv2.cvtColor(np.squeeze(cv2.imread(srnm)), cv2.COLOR_BGR2GRAY)
        else:
            sr = tiff.imread(srnm)
        
        sr = cv2.resize(np.array(sr), (128, 128))
        hr = cv2.resize(np.array(hr), (128, 128))
        # print('lr sr hr.shape = ', lr.shape, sr.shape, hr.shape)
        print(self.filenamesinput[-1], '\n', self.filenames[-1])
        if len(sr.shape) < 3:
            sr = np.expand_dims(sr, -1)
        if len(lr.shape) < 3:
            lr = np.expand_dims(lr, -1)
        if len(hr.shape) < 3:
            hr = np.expand_dims(hr, -1)
                
        lr = normalize(lr, 0, 100, clip=True) * 2 - 1
        sr = normalize(sr, 0, 100, clip=True) * 2 - 1
        hr = normalize(hr, 0, 100, clip=True) * 2 - 1
        srtensor = torch.from_numpy(np.ascontiguousarray(sr.transpose((2, 0, 1)))).float()
        hrtensor = torch.from_numpy(np.ascontiguousarray(hr.transpose((2, 0, 1)))).float()
        lrtensor = torch.from_numpy(np.ascontiguousarray(lr.transpose((2, 0, 1)))).float()
        pair_t = [lrtensor, srtensor, hrtensor]
        return pair_t[0], pair_t[1], pair_t[2], filename

    def __len__(self):
        print('len(self.images_hr)', len(self.filenames))
        return len(self.filenames)


class Flourescenedenoise(data.Dataset):
    def __init__(self, LRpath='', name='Denoising_Planaria', c=1,
                 rootdatapath='/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'):
        self.datamin, self.datamax = 0, 100
        self.denoisegt = name
        self.datapath = rootdatapath + '%s/' % self.denoisegt
        self._scandenoisetif(c, LRpath)
        self.lenth = len(self.nm_lrdenoise)
        print('++ ++ ++ ++ ++ ++ denoisegt %s: self.length of test images = ' % self.denoisegt, self.lenth, '++ ++ ++ ++ ++ ++')
    
    def _scandenoisetif(self, c=1, inputpath=''):
        lr = []
        if ('Planaria' in self.denoisegt) or ('Tribolium' in self.denoisegt):
            lr.extend(sorted(glob.glob(self.datapath + 'test_data/condition_%d/*.tif' % c)))
            self.hrpath = self.datapath + 'test_data/GT/'
        lr.sort()
        self.nm_lrdenoise = lr
        
        ext = '.tif'
        if os.path.exists(inputpath + 'EXP280_Smed_live_RedDot1_slide_mnt_N3_stk3-ZYX_noNorm_noUint8.tif') \
            or os.path.exists(inputpath + 'nGFP_0.1_0.2_0.5_20_30_mid-ZYX_noNorm_noUint8.tif'):
                ext = '-ZYX_noNorm_noUint8.tif'
        elif os.path.exists(inputpath + 'pred_EXP280_Smed_live_RedDot1_slide_mnt_N3_stk3.tif')\
            or os.path.exists(inputpath + 'pred_nGFP_0.1_0.2_0.5_20_30_mid.tif'):
                inputpath = inputpath + 'pred_'
        self.nm_srdenoise = []
        for namepath in self.nm_lrdenoise:
            name, _ = os.path.splitext(os.path.basename(namepath))
            self.nm_srdenoise.append(inputpath + '%s%s' %(name, ext))
        
    def __getitem__(self, idx):
        idx = idx % self.lenth
        lr, sr, hr, filename, d = self._load_file_denoise(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()
        sr = torch.from_numpy(np.ascontiguousarray(sr)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr)).float()
        return lr, sr, hr, filename
    
    def __len__(self):
        return self.lenth
            
    def _load_file_denoise(self, idn):
        filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idn]))
        rgb = np.float32(imread(self.hrpath + filename + fmt))
        rgblr = np.float32(imread(self.nm_lrdenoise[idn]))       
        rgbsr = np.float32(imread(self.nm_srdenoise[idn]))       
        # print('Test Denoise, ----> rgblr.shape', rgblr.shape)  # , rgblr.max(), rgblr.min()
        return rgblr, rgbsr, rgb, filename, self.denoisegt


class Flouresceneiso(data.Dataset):
    def __init__(self, LRpath='', name='Isotropic_Liver',
                rootdatapath='/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Isotropic/'):
        self.datamin, self.datamax = 0, 100
        self.iso = name  #
        self.datapath = rootdatapath + '%s/train_data/' % self.iso
        self.dir_lr = rootdatapath + '%s/test_data/' % self.iso
        self._scaniso()
        self.inputpath = LRpath
        self.lenth = len(self.nm_lr)
        print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        
    def _scaniso(self):
        hr = []
        lr = []
        if self.iso == 'Isotropic_Liver':
            hr.append(self.dir_lr + 'input_subsample_1_groundtruth.tif')
            lr.append(self.dir_lr + 'input_subsample_8.tif')
        else:
            filenames = os.listdir(self.dir_lr)
            for fi in range(len(filenames)):
                name = filenames[fi][:-4]
                lr.append(self.dir_lr + name + '.tif')
        
        self.nm_hr, self.nm_lr = hr, lr
       
    def __getitem__(self, idx):
        idx = idx% self.lenth
        lr, sr, hr, filename = self._load_file_isotest(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()
        sr = torch.from_numpy(np.ascontiguousarray(sr)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr)).float()
        return lr, sr, hr, filename
    
    def __len__(self):
        return self.lenth

    def _load_file_isotest(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        rgblr = np.float32(imread(self.nm_lr[idx]))
        
        if 'Isotropic_Liver' in self.nm_lr[idx]:
            hrp = self.nm_lr[idx].replace('_8.tif', '_1_groundtruth.tif')
            rgb = np.float32(imread(hrp))
            
            sr = np.float32(imread(self.inputpath))
            return rgblr, sr, rgb, filename
        elif 'Retina' in self.nm_lr[idx]:
            rgblr = np.transpose(zoom(rgblr, (10.2, 1, 1, 1), order=1), [0, 2, 3, 1])
            print('ISO Testset ', i, ', ----> rgblr.max()', rgblr.max(), rgblr.shape)
            return rgblr, rgblr, filename


class Flouresceneproj(data.Dataset):
    def __init__(self, LRpath='', name='Projection_Flywing', condition=0,
                 rootdatapath= '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'):
        self.iso = [name]
        self.rootdatapath = rootdatapath + '%s/' % self.iso[0]
        self.inputpath = LRpath
        self._scan(condition)
        self.lenth = len(self.nm_lr)
        print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def _scan(self, condition):
        hr = []
        lr = []
        self.dir_lr = self.rootdatapath + 'test_data/'
        lr.extend(glob.glob(self.dir_lr + 'Input/C%d/*.tif' % condition))
        hr.extend(glob.glob(self.dir_lr + 'GT/C%d/*.tif' % condition))
        # hr.extend(glob.glob(self.dir_lr + 'GT/C2/*.tif'))
        hr.sort()
        lr.sort()
        self.nm_hr, self.nm_lr = hr, lr
        self.ext = '.tif'
        if os.path.exists(self.inputpath + 'pred_C%d_T026.tif' % condition):
            self.inputpath = self.inputpath + 'pred_'  # GVTNet
        elif os.path.exists(self.inputpath + 'my_model_C%d_T026.tif' % condition):
            self.inputpath = self.inputpath + 'my_model_'  # CARE
        elif os.path.exists(self.inputpath + 'proj_C%d_T026.tif' % condition):
            self.inputpath = self.inputpath + 'proj_'  # GT
        elif os.path.exists(self.inputpath + 'C%d_T026-MIP.tif' % condition):
            self.ext = '-MIP.tif'
    
    def __getitem__(self, idx):
        lr, sr, hr, filename = self._load_file_test(idx % self.lenth)
        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()
        sr = torch.from_numpy(np.ascontiguousarray(sr)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr)).float()
        return lr, sr, hr, filename
    
    def __len__(self):
        return self.lenth
    
    def _load_file_test(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))  # C0_T026, .tif
        rgblr = np.float32(imread(self.nm_lr[idx]))
        rgb = np.expand_dims(np.float32(imread(self.nm_hr[idx])), 0)        
        
        name = '%s%s' % (filename, self.ext)
        sr = np.float32(imread(self.inputpath + name))
        if len(sr.shape) < 3:
            sr = np.expand_dims(sr, 0)
        
        return rgblr, sr, rgb, filename

 
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
