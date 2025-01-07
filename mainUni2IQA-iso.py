import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import *
import os
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=3)
    parser.add_argument('--resume', type=int, default=0, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--save', type=str, default='', help='_itefile name to save')
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--pre_train', type=str, default='.', help='pre-trained model directory')
    
    # Data specifications
    parser.add_argument('--data_test', type=str, default=testset, help='demo image directory')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    parser.add_argument('--cpu', action='store_true', default=False, help='')
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    
    parser.add_argument('--scale', type=str, default='1', help='super resolution scale')
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
    
    # Model specifications
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'), help='FP precision for test (single | half)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


class Flouresceneiso(data.Dataset):
    def __init__(self, INpath='', reconstructedpath=''):
        self.datamin, self.datamax = 0, 100
        self.nm_rec = [INpath]
        self.nm_in = [reconstructedpath]
            
    def __getitem__(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_rec[idx]))
        lr = np.float32(imread(self.nm_in[idx]))
        sr = np.float32(imread(self.nm_rec[idx]))
        
        if 'Retina' in self.nm_in[idx]:
            lr = np.transpose(zoom(lr, (10.2, 1, 1, 1), order=1), [0, 2, 3, 1])
        
        # print(lr.shape)  # (301, 752, 752)
        # print(sr.shape)  # (301, 752, 752)
        # lr = lr[:101, :, :]
        # sr = sr[:101, :, :]
        
        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()
        sr = torch.from_numpy(np.ascontiguousarray(sr)).float()
        return lr, sr, filename
    
    def __len__(self):
        return 1


class Trainer():
    def __init__(self, args, loader_test, my_model):
        self.args = args
        gpu = torch.cuda.is_available()
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        self.scale = args.scale
        self.loader_test = loader_test
        self.model = my_model
        self.normalizer = PercentileNormalizer(2, 99.8)  # 逼近npz

    def test(self):
        def _rotate(arr, k=1, axis=1, copy=True):
            """Rotate by 90 degrees around the first 2 axes."""
            if copy:
                arr = arr.copy()
            k = k % 4
            arr = np.rollaxis(arr, axis, arr.ndim)
            if k == 0:
                res = arr
            elif k == 1:
                res = arr[::-1].swapaxes(0, 1)
            elif k == 2:
                res = arr[::-1, ::-1]
            else:
                res = arr.swapaxes(0, 1)[::-1]
            
            res = np.rollaxis(res, -1, axis)
            return res
        datamin, datamax = self.args.datamin, self.args.datamax
        torch.set_grad_enabled(False)
        self.model.eval()
        num = 0
        pslstref = []
        sslstref = []
        nmlst = []
        for idx_data, (lrt, srt, filename) in enumerate(self.loader_test[0]):
            num += 1
            name = '{}'.format(filename[0])
            nmlst.append(name)
            srt = self.normalizer.before(srt, 'CZYX')
            lrt = self.normalizer.before(lrt, 'CZYX')

            srt, lrt = self.prepare(srt, lrt)  # [B, 301, 752, 752]
            sr = np.float32(srt.cpu().detach().numpy())
            sr = np.squeeze(self.normalizer.after(sr))

            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))
            if len(lr.shape) <= 3: lr = np.expand_dims(lr, -1)
            isoim1 = np.zeros_like(lr, dtype=np.float32)  # [301, 752, 752, 2]
            isoim2 = np.zeros_like(lr, dtype=np.float32)
            
            batchstep = 100
            for wp in range(0, lr.shape[2], batchstep):
                if wp + batchstep >= lr.shape[2]:
                    wp = lr.shape[2] - batchstep
                x_rot1 = _rotate(lr[:, :, wp:wp + batchstep, :], axis=1, copy=False)
                x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)
                x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()
                x_rot1 = self.prepare(x_rot1)[0]
                a1 = self.model(x_rot1, 3)
                
                a1 = np.expand_dims(np.squeeze(a1.cpu().detach().numpy()), -1)
                u1 = _rotate(a1, -1, axis=1, copy=False)
                isoim1[:, :, wp:wp + batchstep, :] = u1
                
            for hp in range(0, lr.shape[1], batchstep):
                if hp + batchstep >= lr.shape[1]:
                    hp = lr.shape[1] - batchstep
                
                x_rot2 = _rotate(_rotate(lr[:, hp:hp + batchstep, :, :], axis=2, copy=False), axis=0, copy=False)
                x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)
                x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()
                a2 = self.model(self.prepare(x_rot2)[0], 3)
                
                a2 = np.expand_dims(np.squeeze(a2.cpu().detach().numpy()), -1)
                u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
                isoim2[:, hp:hp + batchstep, :, :] = u2
                                                
            SRGT = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
            SRGT = np.squeeze(self.normalizer.after(SRGT))
            c, h, w = sr.shape
            
            cpsnrlst = []
            cssimlst = []
            for dp in range(1, h, h // 5):
                srpatch = normalize(sr[:, dp, :], datamin, datamax, clip=True) * 255
                SRGTpatch = normalize(SRGT[:, dp, :], datamin, datamax, clip=True) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(srpatch, SRGTpatch)
                print('Patch %s - C%d- PSNR/SSIM = %f/%f' % (name, dp, psm, ssmm))
                cpsnrlst.append(psm)
                cssimlst.append(ssmm)
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            sslstref.append(ssim)
            pslstref.append(psnr1)
        
        psnrmeanref = np.mean(pslstref)
        ssimmeanref = np.mean(sslstref)
        file = open(testsave + "AssHall-PSNRSSIM.txt", 'w')
        file.write('Mean = ' + str(psnrmeanref) + str(ssimmeanref))
        file.write('\nName \n' + str(nmlst)
                    + '\n AssHall(PSNR) \n' + str(pslstref)
                    + '\n AssHall(SSIM) \n' + str(sslstref))
        file.close()
        print(testset, 'num = ', len(self.loader_test[0])
              , 'Mean = ' + str(psnrmeanref) + str(ssimmeanref))

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    

if __name__ == '__main__':
    testset = 'Isotropic_Liver'
    inputpath = ''
    reconstructedpath= ''
    modelpath = './model/checkpoint/isotropic/model_best.pt'
    print('Load Model from ', modelpath)
        
    testsave = './result_IQA/%s/' % (testset)
    os.makedirs(testsave, exist_ok=True)
    
    args = options()
    torch.manual_seed(args.seed)
    unimodel = model.UniModel(args, tsk=3)
    _model = model.Model(args, unimodel, rp='./')
    
    loader_test = [dataloader.DataLoader(
        Flouresceneiso(inputpath, reconstructedpath),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
    
    kwargs = {}    
    _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
    
    t = Trainer(args, loader_test, _model)
    t.test()
