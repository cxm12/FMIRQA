import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import *
import os
import numpy as np
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=2)
    parser.add_argument('--resume', type=int, default=0, help='resume of IQA model')
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


class Flourescenedenoise(data.Dataset):
    def __init__(self, noisepath='', reconstructedpath=''):
        self.datamin, self.datamax = 0, 100

        self.nm_denoise = sorted(glob.glob(reconstructedpath + '/*.tif'))
        self.nm_noise = sorted(glob.glob(noisepath + '/*.tif'))
        self.lenth = len(self.nm_noise)
        
    def __getitem__(self, idx):
        filename, fmt = os.path.splitext(os.path.basename(self.nm_noise[idx]))
        noise = np.float32(imread(self.nm_noise[idx]))       
        denoise = np.float32(imread(self.nm_denoise[idx]))
        # print(noise.shape)  # [depth, H, W]
        # # denoise = denoise[:20, :64, :64]
        # # noise = noise[:20, :64, :64]
        noise = torch.from_numpy(np.ascontiguousarray(noise)).float()
        denoise = torch.from_numpy(np.ascontiguousarray(denoise)).float()
        return noise, denoise, filename
    
    def __len__(self):
        return self.lenth


class Trainer():
    def __init__(self, args, loader_test, my_model):
        self.args = args
        gpu = torch.cuda.is_available()
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        self.scale = args.scale
        self.loader_test = loader_test
        self.model = my_model
        self.normalizer = PercentileNormalizer(2, 99.8)
        
    def test(self):
        file = open(testsave + "AssHall-PSNRSSIM-C%d.txt"% condition, 'w')
        torch.set_grad_enabled(False)
        self.model.eval()
        
        sslst = []
        pslst = []
        nmlst = []
        for idx_data, (noiset, denoiset, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
                        
            denoiset = self.normalizer.before(denoiset, 'CZYX')
            [noiset, denoiset] = self.prepare(noiset, denoiset)
            denoise = np.squeeze(denoiset.cpu().detach().numpy())
            denoise255 = np.float32(normalize(denoise, 0, 100, clip=True)) * 255
            
            noise = np.squeeze(noiset.cpu().detach().numpy())
            denoiseimGT = torch.zeros_like(noiset, dtype=noiset.dtype)
            
            batchstep = 5  # 10  #
            inputlst = []
            for ch in range(0, len(noise)):
                if ch < 5 // 2:
                    noise1 = [noiset[:, ch:ch + 1, :, :] for _ in range(5 // 2 - ch)]
                    noise1.append(noiset[:, :5 // 2 + ch + 1])
                    noiset1 = torch.concat(noise1, 1)  # [B, inputchannel, h, w]
                elif ch >= (len(noise) - 5 // 2):
                    noise1 = []
                    noise1.append(noiset[:, ch - 5 // 2:])
                    numa = (5 // 2 - (len(noise) - ch)) + 1
                    noise1.extend([noiset[:, ch:ch + 1, :, :] for _ in range(numa)])
                    noiset1 = torch.concat(noise1, 1)  # [B, inchannel, h, w]
                else:
                    noiset1 = noiset[:, ch - 5 // 2:ch + 5 // 2 + 1]
                assert noiset1.shape[1] == 5
                inputlst.append(noiset1)
            
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(noise):
                    dp = len(noise) - batchstep
                noisetn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inchannel, h, w]
                a = self.model(noisetn, 2)
                denoiseimGT[:, dp:dp + batchstep, :, :] = torch.transpose(a, 1, 0)  # [1, batch, h, w]
            
            denoiseimGT = np.float32(denoiseimGT.cpu().detach().numpy())
            denoiseimGT = np.squeeze(self.normalizer.after(denoiseimGT))
            denoiseGT255 = np.float32(normalize(denoiseimGT, 0, 100, clip=True)) * 255
            
            cplst = []
            cslst = []
            for dp in range(0, len(inputlst), batchstep):
                psm, ssmm = utility.compute_psnr_and_ssim(denoise255[dp], denoiseGT255[dp])
                cplst.append(psm)
                cslst.append(ssmm)
            pslst.append(np.mean(cplst))
            sslst.append(np.mean(cslst))
        
        psnrmeanref = np.mean(pslst)
        ssimmeanref = np.mean(sslst)
        print(psnrmeanref, ssimmeanref)
        file = open(testsave + "C%d.txt" % (condition), 'w')
        file.write('\n \n +++++++++ condition%d ++++++++++++ \n' % (condition))
        file.write('Mean = ' + str(psnrmeanref) + str(ssimmeanref))
        file.write('\nName \n' + str(nmlst)
                    + '\n AssHall(PSNR) \n' + str(pslst)
                    + '\n AssHall(SSIM) \n' + str(sslst))
        file.close()
        print(testset, '+++++++++ condition%d++++++++++++' % condition, 'num = ', len(self.loader_test[0]) 
              , 'Mean = ' + str(psnrmeanref) + str(ssimmeanref))

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        return [_prepare(a) for a in args]


if __name__ == '__main__':
    condition = 2
    for testset in ['Tribolium', 'Planaria']:
        denoisepath = ''
        noisepath = ''
        modelpath = './model/checkpoint/denoise/%s/model_best.pt' % testset
    
        testsave = './result_IQA/%s/' % testset
        os.makedirs(testsave, exist_ok=True)

        args = options()
        torch.manual_seed(args.seed)
        unimodel = model.UniModel(args, tsk=2)
        _model = model.Model(args, unimodel, rp='./')
        loader_test = [dataloader.DataLoader(
            Flourescenedenoise(noisepath=noisepath, reconstructedpath=denoisepath),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
        
        kwargs = {}
        _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
        print('Load Model from ', modelpath)
        
        t = Trainer(args, loader_test, _model)
        t.test()
