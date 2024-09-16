import os
import torch
import torch.nn as nn
from model.attention import ProjectionUpdater
from model.Unimodel import UniModel
gpu = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, args, srmodel=None, rp='.'):
        super(Model, self).__init__()
        print('Making model...')
        self.scale = args.scale[0]
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if (not gpu) else 'cuda')
        print(self.device, ' = self.device')
        
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        self.outchannel = args.n_colors
        self.model = srmodel.to(self.device)
        
        self.proj_updater = ProjectionUpdater(self.model, feature_redraw_interval=640)
        if args.precision == 'half':
            self.model.half()

        if args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        
        if args.pre_train == '.':
            self.dir = os.path.join(rp, 'experiment', args.save)
            
        print('self.dir = ', self.dir)
        os.makedirs(self.dir, exist_ok=True)
        self.load(self.dir, pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)
        # print(self.model)
        os.makedirs(self.dir, exist_ok=True)

    def forward(self, x, tsk, hr_coord=None, cell=None):
        self.tsk = tsk
        self.proj_updater.redraw_projections()
        if cell is not None:
            return self.model(x, hr_coord, cell)
        elif self.chop and not self.training:
            if tsk == 4:
                return self.forward_chopProj(x)
            elif tsk == 5:
                return self.forward_chop2to3(x)
            else:
                return self.forward_chop(x)
        else:
            return self.model(x)

    def state_dict(self, **kwargs):
        target = self.model
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.model
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch)))

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        kwargs = {}

        if resume == -1:
            print('Load Model from ', os.path.join(apath, 'model_latest.pt'))
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=True
            )
        elif resume == -2:
            m = os.path.join(apath, 'model_best.pt')
            print('Load Model from ', m)
            try:
                self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
            except:
                print('Load Model from ', os.path.join(apath, 'model_latest.pt'))
                self.model.load_state_dict(
                    torch.load(os.path.join(apath, 'model_latest.pt'),**kwargs), strict=True)
        elif resume < -2:
            m = os.path.join(apath, 'model_best%d.pt' % -resume)
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume == 0 and pre_train != '.':
            print('Loading pre_train UNet model from {}'.format(pre_train))
            if os.path.exists(pre_train):
                self.model.load_state_dict(torch.load(pre_train, **kwargs), strict=False)
            else:
                print('No pretrain model from {}'.format(pre_train))
                # exit()
        elif resume > 0:  # and os.path.exists(os.path.join(apath, 'model_{}.pt'.format(resume))):
            print('Load Model from ', os.path.join(apath, 'model_{}.pt'.format(resume)))
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=True
            )
        else:
            print('!!!!!!!!  Not Load Model  !!!!!!')
            assert resume == 0 and pre_train == '.'

    def forward_chop(self, x, shave=10, min_size=40000, cell=None, hr_coord=None):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, _, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + 16, w_half + 16
        h_size += 8
        w_size += 8

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size)\
                for patch in lr_list]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        
        output = x.new(b, self.outchannel, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_chopProj(self, x, shave=10, min_size=120000):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += 4-h_size % 4
        w_size += 8-w_size % 8
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
    
        if w_size * h_size < min_size:
            sr_list = []
            sr_lists1 = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # print('Proj Output')
                sr_batch1, sr_batch = self.model(lr_batch)
                # sr_batch1, sr_batch, _ = self.model(lr_batch)
                sr_lists1.extend(sr_batch1.chunk(n_GPUs, dim=0))
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_lists1 = [self.forward_chopProj(patch, shave=shave, min_size=min_size)[0] for patch in lr_list]
            sr_list = [
                self.forward_chopProj(patch, shave=shave, min_size=min_size)[1] for patch in lr_list]
    
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
    
        output = x.new(b, 1, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        outputs1 = x.new(b, 1, h, w)
        outputs1[:, :, 0:h_half, 0:w_half] = sr_lists1[0][:, :, 0:h_half, 0:w_half]
        outputs1[:, :, 0:h_half, w_half:w] = sr_lists1[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        outputs1[:, :, h_half:h, 0:w_half] = sr_lists1[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        outputs1[:, :, h_half:h, w_half:w] = sr_lists1[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return outputs1, output

    def forward_chop2to3(self, x, shave=2, min_size=120000):
        scale = 11
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + shave, w_half + shave
      
        h_size += 8-h_size % 8
        w_size += 8-w_size % 8
        
        print('0 x.size = ', x.size(), 'h/w_size = ', h_size, w_size)
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
    
        if w_size * h_size < min_size:
            sr_list = []
            sr_lists1 = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batchu, sr_batch = self.model(lr_batch)
                sr_lists1.extend(sr_batchu.chunk(n_GPUs, dim=0))
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_lists1 = [self.forward_chop2to3(patch, shave=shave, min_size=min_size)[0] for patch in lr_list]
            sr_list = [self.forward_chop2to3(patch, shave=shave, min_size=min_size)[1] for patch in lr_list]
    
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half  # 88, 88
        h_size, w_size = scale * h_size, scale * w_size  # 352 352

        output = x.new(b, 61, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        outputs1 = x.new(b, 61, h, w)
        outputs1[:, :, 0:h_half, 0:w_half] \
            = sr_lists1[0][:, :, 0:h_half, 0:w_half]
        outputs1[:, :, 0:h_half, w_half:w] \
            = sr_lists1[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        outputs1[:, :, h_half:h, 0:w_half] \
            = sr_lists1[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        outputs1[:, :, h_half:h, w_half:w] \
            = sr_lists1[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
        return outputs1, output
