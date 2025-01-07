import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import *
import os
import math
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=4)
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


class Flouresceneproj(data.Dataset):
    def __init__(self, inputpath, reconstructedpath):
        self.nm_lr = glob.glob(inputpath + '*.tif')
        self.nm_rec = glob.glob(reconstructedpath + '*.tif')

        self.lenth = len(self.nm_lr)
        print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        
    def __getitem__(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        lr = np.float32(imread(self.nm_lr[idx]))
        rec = np.expand_dims(np.float32(imread(self.nm_rec[idx])), 0)

        if len(rec.shape) < 3:
            rec = np.expand_dims(rec, 0)
                
        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()
        rec = torch.from_numpy(np.ascontiguousarray(rec)).float()
        return lr, rec, filename
    
    def __len__(self):
        return self.lenth
    
    
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
        
    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        datamin, datamax = self.args.datamin, self.args.datamax
        pslstref = []
        sslstref = []
        nmlst = []
        for idx_data, (lrt, srt, filename) in enumerate(self.loader_test[0]):
            name = '{}'.format(filename[0])
            nmlst.append(name)
            lrt, srt = self.prepare(lrt, srt)
            b, c, h, w = srt.shape
            
            _, SRGTt = self.model(lrt.expand(b, 50, h, w), 4)

            sr = np.float32(np.squeeze(srt.cpu().detach().numpy()))
            SRGT = np.float32(np.squeeze(SRGTt.cpu().detach().numpy()))
            
            sr255 = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255
            SRGT255 = np.float32(normalize(SRGT, datamin, datamax, clip=True)) * 255    
                    
            ps255ref, ss255ref = utility.compute_psnr_and_ssim(sr255, SRGT255)
            pslstref.append(ps255ref)
            sslstref.append(ss255ref)
            
        psnrmeanref = np.mean(pslstref)
        ssimmeanref = np.mean(sslstref)
        print(testset, 'num = ', len(self.loader_test[0]),
              '+++++++++ condition %d ++++++++++++' % condition, psnrmeanref, ssimmeanref)
        file = open(testsave + "AssHall-PSNRSSIM-C%d.txt" % condition, 'w')
        file.write('Mean = ' + str(psnrmeanref) + str(ssimmeanref))
        file.write('\nName \n' + str(nmlst)
                    + '\n AssHall(PSNR) \n' + str(pslstref)
                    + '\n AssHall(SSIM) \n' + str(sslstref))
        file.close()
        torch.set_grad_enabled(True)
    
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    

if __name__ == '__main__':
    testset = 'Projection_Flywing'
    for condition in range(0, 4):
        inputpath = ''
        reconstructedpath = ''
        
        modelpath = './model/checkpoint/Projection/model_best.pt'
    
        testsave = './result_IQA/%s/' % testset
        os.makedirs(testsave, exist_ok=True)

        args = options()
        torch.manual_seed(args.seed)
        unimodel = model.UniModel(args, tsk=4)
        _model = model.Model(args, unimodel, rp='./')
        loader_test = [dataloader.DataLoader(
            Flouresceneproj(inputpath, reconstructedpath),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]
        
        kwargs = {}        
        print('Load Model from ', modelpath)
        _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
    
        t = Trainer(args, loader_test, args.data_test, _model)
        t.test()
