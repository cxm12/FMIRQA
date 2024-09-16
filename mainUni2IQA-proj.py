import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import np, normalize, PercentileNormalizer, Flouresceneproj
import os
from utility import savecolorim
import math
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--resume', type=int, default=0, help='')
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
    modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIRProjection_Flywing/testevery1/server2/P128B1/model_best59.pt'
    print('Load Model from ', modelpath)
    _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
    
    
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
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        
    # # -------------------------- Projection --------------------------
    def testproj(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        datamin, datamax = self.args.datamin, self.args.datamax
        pslstref = []
        sslstref = []
        nmlst = []
        axes_restored = 'YX'
        for idx_data, (lrt, srt, hrt, filename) in enumerate(self.loader_test[0]):
            name = '{}'.format(filename[0])
            nmlst.append(name)
            [srt] = self.prepare(srt)
            b, c, h, w = srt.shape
            SR_sr_stg1, SR_srt = self.model(srt.expand(b, 50, h, w), task)

            sr = np.float32(np.squeeze(srt.cpu().detach().numpy()))
            SR_sr = np.float32(np.squeeze(SR_srt.cpu().detach().numpy()))
            utility.save_tiff_imagej_compatible(testsave + name + '-SR_SR.tif', SR_sr, axes_restored)
            
            sr255 = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255
            SR_sr255 = np.float32(normalize(np.float32(SR_sr), datamin, datamax, clip=True)) * 255            
            ps255ref, ss255ref = utility.compute_psnr_and_ssim(sr255, SR_sr255)
            # print('2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, ps255ref, ss255ref))
            # savecolorim(testsave + name[:-4] + '-SR_SR.png', SR_sr, norm=False)
            savecolorim(testsave + name[:-4] + '-df_Input_SR_SR.png', np.clip(np.abs(SR_sr255 - sr255), 0, 255), norm=False)
            
            if math.isinf(ps255ref): ps255ref = 100
            pslstref.append(ps255ref)
            sslstref.append(ss255ref)
            
        psnrmeanref = np.mean(pslstref)
        ssimmeanref = np.mean(sslstref)
        file = open(testsave + "Psnrssim_RefSR_of_SR_UniFMIR_c%d.txt" % condition, 'w')
        file.write('Mean between input and SR(input) = ' + str(psnrmeanref) + str(ssimmeanref))
        file.write('\nName \n' + str(nmlst) 
                    + '\n PSNR between input and SR(input) \n' + str(pslstref)
                    + '\n SSIM \n' + str(sslstref))
        file.close()
        print(testset, 'num = ', len(self.loader_test[0]),
              '+++++++++ condition %d ++++++++++++' % condition, psnrmeanref, ssimmeanref)
        torch.set_grad_enabled(True)
    
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    

if __name__ == '__main__':
    task = 4
    testset = 'Projection_Flywing'
    for condition in range(2, 4):
        inputpathGT = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/%s/test_data/GT/C%d/' % (testset, condition)
        method = 'CARE'
        inputpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/projection/results/C%d/' % (condition)
        testsave = './result_IQA/task%d_%s/%s/C%d/' % (task, testset, method, condition)
        os.makedirs(testsave, exist_ok=True)

        args = options()
        torch.manual_seed(args.seed)
        unimodel = model.UniModel(args, tsk=task)
        _model = model.Model(args, unimodel, rp='./')
        loader_test = [dataloader.DataLoader(
            Flouresceneproj(LRpath=inputpath, name=testset, condition=condition),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
        loadUniFMIRgpu()
        t = Trainer(args, loader_test, args.data_test, _model)
        t.testproj(condition)
