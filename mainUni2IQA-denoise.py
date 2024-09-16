import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import imsave, normalize, PercentileNormalizer, Flourescenedenoise
from utility import savecolorim
import os
import numpy as np
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=task)
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


def loadUniFMIRgpu():
    kwargs = {}
    if testset == 'Denoising_Planaria':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIR%s/testevery1/P64B16/model_best100.pt' % testset
    if testset == 'Denoising_Tribolium':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/'\
                    'experiment/Uni-SwinIR%s/server2/testevery1/P64B16/Ep101_data10/model_best.pt' % testset
    _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
    print('Load Model from ', modelpath)


class Trainer():
    def __init__(self, args, loader_test, datasetname, my_model):
        self.args = args
        gpu = torch.cuda.is_available()
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        self.scale = args.scale
        self.datasetname = datasetname
        self.loader_test = loader_test
        self.model = my_model
        self.normalizer = PercentileNormalizer(2, 99.8)  # 逼近npz
        
    # # -------------------------- 3D denoise --------------------------
    def test3Ddenoise(self, data_test='Denoising_Tribolium'):
        file = open(testsave + "Psnrssim_RefUniFMIR_c%d.txt"% condition, 'w')
        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        torch.set_grad_enabled(False)
        self.model.eval()
        
        sslst = []
        pslst = []
        nmlst = []
        for idx_data, (_, srt, _, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            name = filename[0]
                        
            srt = self.normalizer.before(srt, 'CZYX')  # [0~806] -> [0~1.]
            [srt] = self.prepare(srt)
            sr = np.squeeze(srt.cpu().detach().numpy())
            denoiseim = torch.zeros_like(srt, dtype=srt.dtype)
            
            batchstep = 5  # 10  #
            inputlst = []
            for ch in range(0, len(sr)):  # [45, 486, 954]  0~44
                if ch < 5 // 2:  # 0, 1
                    sr1 = [srt[:, ch:ch + 1, :, :] for _ in range(5 // 2 - ch)]
                    sr1.append(srt[:, :5 // 2 + ch + 1])
                    srt1 = torch.concat(sr1, 1)  # [B, inputchannel, h, w]
                elif ch >= (len(sr) - 5 // 2):  # 43, 44
                    sr1 = []
                    sr1.append(srt[:, ch - 5 // 2:])
                    numa = (5 // 2 - (len(sr) - ch)) + 1
                    sr1.extend([srt[:, ch:ch + 1, :, :] for _ in range(numa)])
                    srt1 = torch.concat(sr1, 1)  # [B, inputchannel, h, w]
                else:
                    srt1 = srt[:, ch - 5 // 2:ch + 5 // 2 + 1]
                assert srt1.shape[1] == 5
                inputlst.append(srt1)
            
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(sr):
                    dp = len(sr) - batchstep
                # print(dp)  # 0, 10, .., 90
                srtn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inputchannel, h, w]
                a = self.model(srtn, task)
                a = torch.transpose(a, 1, 0)  # [1, batch, h, w]
                denoiseim[:, dp:dp + batchstep, :, :] = a
            
            SR_sr = np.float32(denoiseim.cpu().detach().numpy())
            sr = np.squeeze(self.normalizer.after(sr))
            SR_sr = np.squeeze(self.normalizer.after(SR_sr))
            imsave(testsave + name + '-SR_SR.tif', SR_sr)
            
            sr255 = np.float32(normalize(sr, 0, 100, clip=True)) * 255
            SR_sr255 = np.float32(normalize(SR_sr, 0, 100, clip=True)) * 255
                        
            cpsnrlst = []
            cssimlst = []            
            step = 1
            if 'Planaria' in data_test:
                if condition == 1:
                    randcs = 10
                    randce = sr.shape[0] - 10
                    step = (sr.shape[0] - 20) // 5
                else:
                    randcs = 85
                    randce = 87
                    step = 1
                    if randce >= sr.shape[0]:
                            randcs = sr.shape[0] - 3
                            randce = sr.shape[0]

                for dp in range(randcs, randce, step):
                        savecolorim(testsave + name + '-SR_SRD%d.png' % dp, SR_sr[dp], norm=False)
                        SR_sr2 = np.round(np.maximum(0, np.minimum(255, SR_sr[dp])))
                        sr2 = np.round(np.maximum(0, np.minimum(255, sr[dp])))
                        savecolorim(testsave + name + '-df_Input_SR_SRD%d.png' % dp, 
                                    np.clip(np.abs(SR_sr2 - sr2), 0, 255), norm=False)
            
                        srpatch255 = sr255[dp, :patchsize, :patchsize]
                        SR_srpatch255 = SR_sr255[dp, :patchsize, :patchsize]
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch255, SR_srpatch255)
                        # print('SR Image %s - C%d- PSNR/SSIM = %f/%f' % (name, dp, psm, ssmm))
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                        
            elif 'Tribolium' in data_test:
                    if condition == 1:
                        randcs = 2
                        randce = sr.shape[0] - 2
                        step = (sr.shape[0] - 4) // 6
                    else:
                        randcs = sr.shape[0] // 2 - 1
                        randce = randcs + 3
                        step = 1
                    
                    for randc in range(randcs, randce, step):
                        savecolorim(testsave + name + '-SR_SRD%d.png' % randc, SR_sr[randc], norm=False)
                        SR_sr2 = np.round(np.maximum(0, np.minimum(255, SR_sr[randc])))
                        sr2 = np.round(np.maximum(0, np.minimum(255, sr[randc])))
                        savecolorim(testsave + name + '-df_Input_SR_SRD%d.png' % randc, 
                                    np.clip(np.abs(SR_sr2 - sr2), 0, 255), norm=False)
                        
                        SR_srpatch255 = normalize(SR_sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        srpatch255 = normalize(sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch255, SR_srpatch255)
                        # print('SR Image %s - C%d- PSNR/SSIM = %f/%f' % (name, randc, psm, ssmm))
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)                     
            
            sslst.append(np.mean(np.array(cssimlst)))
            pslst.append(np.mean(np.array(cpsnrlst)))
        
        psnrmeanref = np.mean(pslst)
        ssimmeanref = np.mean(sslst)
        print(psnrmeanref, ssimmeanref)
        file = open(testsave + "%sC%dPsnrssim_RefSR_of_SR_UniFMIR.txt" % (method, condition), 'w')
        file.write('\n \n +++++++++ condition%d meanSR ++++++++++++ \n PSNR/SSIM \n  patchsize = %d \n' % (
                condition, patchsize))
        file.write('Mean between input and SR(input) = ' + str(psnrmeanref) + str(ssimmeanref))
        file.write('\nName \n' + str(nmlst)
                    + '\n PSNR between input and SR(input) \n' + str(pslst)
                    + '\n SSIM \n' + str(sslst))
        file.close()
        print(testset, '+++++++++ condition%d++++++++++++' % condition, 'num = ', len(self.loader_test[0]) 
              , '\n ssimmeanref/psnrmeanref = ', psnrmeanref, ssimmeanref)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        return [_prepare(a) for a in args]


if __name__ == '__main__':
    task = 2
    condition = 2
    method = 'CARE'
    for testset in ['Denoising_Planaria']:  # 'Denoising_Tribolium', 
        inputpathGT = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/%s/test_data/GT/' % (testset)
        inputpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/denoising2D/models/epoch200/my_model/%s/result/Norm_0-100/condition_%d/' % (testset, condition)

        testsave = './result_IQA/task%d_%s/%s/C%d/' % (task, testset, method, condition)
        os.makedirs(testsave, exist_ok=True)

        args = options()
        torch.manual_seed(args.seed)
        unimodel = model.UniModel(args, tsk=task)
        _model = model.Model(args, unimodel, rp='./')
        loader_test = [dataloader.DataLoader(
            Flourescenedenoise(LRpath=inputpath, name=testset, c=condition),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
        loadUniFMIRgpu()
        t = Trainer(args, loader_test, args.data_test, _model)
        t.test3Ddenoise()
