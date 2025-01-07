import faulthandler
faulthandler.enable()
import torch
import utility
torch.backends.cudnn.enabled = False
import argparse
from mydataIQA import *
from torch.utils.data import dataloader
import model
import os
# from torchvision.transforms import Resize


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--task', type=int, default=1)
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


class SR(data.Dataset):
    def __init__(self, LRpath='', SRpath=''):
        self.nm_sr = sorted(glob.glob(SRpath + '/*.tif'))
        self.nm_lr = sorted(glob.glob(LRpath + '/*.tif'))
        self.lenth = len(self.nm_lr)

        for namepath in self.nm_sr:
            self.name, _ = os.path.splitext(os.path.basename(namepath))

    def getitem_IQA(self, idx):
        lrnm, srnm, filename = self.nm_lr[idx], self.nm_sr[idx], self.name[idx]
        lr = tiff.imread(lrnm)
        sr = tiff.imread(srnm)

        if len(sr.shape) < 3:
            sr = np.expand_dims(sr, -1)
        if len(lr.shape) < 3:
            lr = np.expand_dims(lr, -1)
                        
        lr = normalize(lr, 0, 100, clip=True) * 2 - 1
        sr = normalize(sr, 0, 100, clip=True) * 2 - 1
        srtensor = torch.from_numpy(np.ascontiguousarray(sr.transpose((2, 0, 1)))).float()
        lrtensor = torch.from_numpy(np.ascontiguousarray(lr.transpose((2, 0, 1)))).float()
        pair_t = [lrtensor, srtensor]
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        return len(self.name)

   
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
        self.model.scale = 2
        torch.set_grad_enabled(False)
        self.model.eval()

        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, sr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            lr, sr = self.prepare(lr, sr)
            SR_lr = self.model(lr, 1)
            # resize_transform = Resize(size=256)
            # sr = resize_transform(sr)
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
            
        psnrmean = np.mean(pslst)
        ssimmean = np.mean(sslst)
        file = open(testsave + "AssHall-PSNRSSIM.txt", 'w')
        file.write('Mean AssHall(PSNR/SSIM)= ' + str(psnrmean) + '/' + str(ssimmean))
        file.write('\nName \n' + str(nmlst) + '\nAssHall(PSNR) \n' + str(pslst)
                    + '\nAssHall(SSIM) \n' + str(sslst))
        file.close()
        print(testset, 'num = ', len(self.loader_test[0]), '\n Mean AssHall(PSNR/SSIM)= ', psnrmean, ssimmean)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    

if __name__ == '__main__':
    testset = 'Microtubules'  # 'ER'  # 'F-actin'  # 'CCPs'  #
    inputpath = ''
    reconstructpath = ''
    
    testsave = './result_IQA/%s/' % testset
    os.makedirs(testsave, exist_ok=True)
    args = options()
    torch.manual_seed(args.seed)
    unimodel = model.UniModel(args, tsk=1)
    _model = model.Model(args, unimodel, rp='./')
    loader_test = [dataloader.DataLoader(
            SR(inputpath, reconstructpath),
            batch_size=1, shuffle=False, pin_memory=True, num_workers=0)]

    kwargs = {}
    modelpath = './model/checkpoint/SR/%s/model_best.pt' % testset
    print('Load Model from ', modelpath)
    _model.model.load_state_dict(torch.load(modelpath, **kwargs), strict=True)

    t = Trainer(args, loader_test, args.data_test, _model)
    t.test()
