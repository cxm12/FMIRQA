import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import np, normalize, PercentileNormalizer, SR
from torch.utils.data import dataloader
import model
import os
from utility import savecolorim
from torchvision.transforms import Resize


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


def get_data_loader():
    loader_test = [dataloader.DataLoader(
            SR(LRpath=inputpath, name=testset),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
    
    return loader_test


def loadUniFMIRgpu():
    kwargs = {}
    if testset == 'Microtubules':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIR%s/Ep101_data10/model_best.pt' % testset
    if testset == 'CCPs':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIR%s/Ep101_data10/model_best.pt' % testset
    if testset == 'F-actin':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIR%s/testevery1/P128B4/Ep101_data10/model_best.pt' % testset
    if testset == 'ER':
        modelpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/BioSR/ENLCA/Uni-FMIR/experiment/Uni-SwinIR%s/server2/P128B1/model_best90.pt' % testset

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
        
    # # -------------------------- SR --------------------------
    def testSR_LR_PSNR(self):
        # PSNR between input and SR(LR)
        self.model.scale = 2
        torch.set_grad_enabled(False)
        self.model.eval()

        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, sr, hr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            lr, sr = self.prepare(lr, sr)
            SR_lr = self.model(lr, task)
            # image_tensor = torch.randn(3, 244, 244)  # 假设的图像tensor，大小为244x244
            
            resize_transform = Resize(size=256)  # 将图像resize到128x128
            if method == 'Input':
                sr = resize_transform(sr)
                
            SR_lr = utility.quantize(SR_lr, self.args.rgb_range)
            sr = utility.quantize(sr, self.args.rgb_range)
            sr = sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            SR_lr = SR_lr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            sr255 = np.float32(normalize(sr, 0, 100, clip=True)) * 255
            SR_lr255 = np.float32(normalize(SR_lr, 0, 100, clip=True)) * 255
            ps255, ss255 = utility.compute_psnr_and_ssim(sr255, SR_lr255)
            pslst.append(ps255)
            sslst.append(ss255)
            print('name %s, ps255, ss255 = ' % filename[0], ps255, ss255)
            
            name = filename[0]
            savecolorim(testsave + name[:-4] + '.png', SR_lr, norm=False)
            SR_lr2 = np.round(np.maximum(0, np.minimum(255, SR_lr)))
            sr2 = np.round(np.maximum(0, np.minimum(255, sr)))
            savecolorim(testsave + name[:-4] + '-df_Input_SRlr.png', np.clip(np.abs(SR_lr2 - sr2), 0, 255), norm=False)
        psnrmean = np.mean(pslst)
        ssimmean = np.mean(sslst)
        file = open(testsave + "Psnrssim_RefSR_of_LR_UniFMIR.txt", 'w')
        file.write('Mean between input and SR of LR = ' + str(psnrmean) + str(ssimmean))
        file.write('\nName \n' + str(nmlst) + '\nPSNR between input and SR of LR \n' + str(pslst)
                    + '\nSSIM \n' + str(sslst))
        file.close()
        print(testset, 'num = ', len(self.loader_test[0]), '\n ssimmean/psnrmean = ', psnrmean, ssimmean)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    

if __name__ == '__main__':
    task = 1
    testset = 'Microtubules'  # 'ER'  # 'F-actin'  # 'CCPs'  #
    inputpathHR = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/GT/' % testset
    inputpath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/output_DFCAN-SISR/' % testset
    method = 'DFCAN'

    testsave = './result_IQA/task%d_%s/%s/' % (task, testset, method)
    os.makedirs(testsave, exist_ok=True)
    args = options()
    torch.manual_seed(args.seed)
    unimodel = model.UniModel(args, tsk=task)
    _model = model.Model(args, unimodel, rp='./')
    loader_test = get_data_loader()
    loadUniFMIRgpu()
    t = Trainer(args, loader_test, args.data_test, _model)
    t.testSR_LR_PSNR()
