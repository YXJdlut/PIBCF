import argparse
import datetime
import logging
import os
import random

from test2 import val_segformer_IJCAI
#设置使用的gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.model_fusion import Network_IJCAI as Network
from datasets import voc_fusion as voc
from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW
from omegaconf import OmegaConf
from architect_average import Architect as Architect1
from core.NewFusionNet import FusionNet

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
#创建随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
#日志设置
def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


#计算预计完成时间
def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)
#验证函数
def validate(model=None, criterion=None, data_loader=None):


    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(
                torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score
from core import Total_fusion_loss, Total_fusion_loss2, Total_fusion_loss3, Fusionloss3, Fusionloss_grad2,Fusionloss4

parser1 = argparse.ArgumentParser("ruas")
parser1.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser1.add_argument('--batch_size', type=int, default=4, help='batch size')
parser1.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser1.add_argument('--learning_rate_min', type=float, default=0.000000001, help='min learning rate')
parser1.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser1.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser1.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser1.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser1.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser1.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser1.add_argument('--layers', type=int, default=8, help='total number of layers')
parser1.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser1.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser1.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser1.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser1.add_argument('--save', type=str, default='EXP', help='experiment name')
parser1.add_argument('--seed', type=int, default=2, help='random seed')
parser1.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser1.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser1.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser1.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
parser1.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args1 = parser1.parse_args()

def train(cfg):

    num_workers = 0
#训练设备初始化
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
#数据加载
    train_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              #shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True)
    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device = torch.device('cuda:0')
#损失函数定义
    # device  =torch.device(args.local_rank)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion_seg = criterion.to(device)
    criterion_total = Fusionloss4().to(device)
    #构建模型与优化器
    model = Network(criterion_total,criterion_seg,cfg.exp.backbone,cfg.dataset.num_classes,256,True).cuda()
    architect = Architect1(model, args1)
    param_groups = model.denoise_net.get_param_groups()
    # model.load_state_dict(torch.load("./checkpoint/fusion_model_08-21-22-10.pth"),strict=False)
    # print('load_pretrained-----------------------------------')
    model.load_state_dict(
    torch.load("./checkpoint/model_Ours_loss4_Bi2_.pth", map_location=device),
    strict=False
    )
    print('load_pretrained-----------------------------------')

    model.to(device)
#优化器
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": model.discriminator.parameters(),
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)
    miou = 0
    flag = False
    miou = val_segformer_IJCAI(model,0,'./0301.txt')
    for n_iter in range(cfg.train.max_iters):
        if (n_iter)%10000 ==0 and flag:
            # "./checkpoint/model" + "_ijcai_fusion_best.pth"
            model.load_state_dict(torch.load("./checkpoint/model" + "_testab1_2mri_BI3.pth"))#重新加载
            flag = False
        try:
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)
            _, inputs_ir_val, inputs_vis_val, inputs_mask_val, labels_val = next(val_loader_iter)
        except:
            # train_sampler.set_epoch(n_iter)

            train_loader_iter = iter(train_loader)
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)
            val_loader_iter = iter(val_loader)
            _, inputs_ir_val, inputs_vis_val, inputs_mask_val, labels_val = next(val_loader_iter)
#输入数据
        inputs_ir = inputs_ir.to(device, non_blocking=True)
        inputs_vis = inputs_vis.to(device, non_blocking=True)
        inputs_mask = inputs_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs_ir = inputs_ir[:,0:1,:,:]
        inputs_mask = inputs_mask[:,0:1,:,:]
        if (n_iter+1)<=2000:
            if (n_iter + 1) % 2 == 0:
                inputs_ir_val = inputs_ir_val[:, 0:1, :, :]
                inputs_mask_val = inputs_mask_val[:, 0:1, :, :]
                inputs_ir_val = inputs_ir_val.to(device, non_blocking=True)
                inputs_vis_val = inputs_vis_val.to(device, non_blocking=True)
                inputs_mask_val = inputs_mask_val.to(device, non_blocking=True)
                labels_val = labels_val.to(device, non_blocking=True)
                architect.step(inputs_ir, inputs_vis, inputs_mask,labels, inputs_ir_val,
                               inputs_vis_val, inputs_mask_val, labels_val, 0.1, unrolled=True,
                               lr_new= optimizer.param_groups[0]['lr']*0.001)
        elif (n_iter+1)<=6000:
            if (n_iter + 1) % 5 == 0:
                inputs_ir_val = inputs_ir_val[:, 0:1, :, :]
                inputs_mask_val = inputs_mask_val[:, 0:1, :, :]
                inputs_ir_val = inputs_ir_val.to(device, non_blocking=True)
                inputs_vis_val = inputs_vis_val.to(device, non_blocking=True)
                inputs_mask_val = inputs_mask_val.to(device, non_blocking=True)
                labels_val = labels_val.to(device, non_blocking=True)
                architect.step(inputs_ir, inputs_vis, inputs_mask,labels, inputs_ir_val,
                               inputs_vis_val, inputs_mask_val, labels_val, 0.1, unrolled=True,
                               lr_new=optimizer.param_groups[0]['lr']*0.001)
        else:
            if (n_iter + 1) % 10 == 0:
                inputs_ir_val = inputs_ir_val[:, 0:1, :, :]
                inputs_mask_val = inputs_mask_val[:, 0:1, :, :]
                inputs_ir_val = inputs_ir_val.to(device, non_blocking=True)
                inputs_vis_val = inputs_vis_val.to(device, non_blocking=True)
                inputs_mask_val = inputs_mask_val.to(device, non_blocking=True)
                labels_val = labels_val.to(device, non_blocking=True)
                architect.step(inputs_ir, inputs_vis, inputs_mask,labels, inputs_ir_val,
                               inputs_vis_val, inputs_mask_val, labels_val, 0.1, unrolled=True,
                               lr_new=optimizer.param_groups[0]['lr']*0.001)
      #损失计算与优化
        optimizer.zero_grad()
        if (n_iter + 1) <= 6000:            
            seg_loss = model._loss(inputs_ir, inputs_vis, inputs_mask, labels)

        else:
            seg_loss = model._loss(inputs_ir, inputs_vis, inputs_mask, labels)
        seg_loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, seg_loss.item()))
        if (n_iter + 1) % 50 == 0:
             with torch.no_grad():
                 #可视化输出
                 fused,  fused_seg =model(inputs_ir,inputs_vis)
                 torchvision.utils.save_image(inputs_ir[:2], 'input_ir_1.png')
                 torchvision.utils.save_image(inputs_vis[:2], 'input_vis_1.png')
                 torchvision.utils.save_image(fused[:2], 'output_1.png')
        # if (n_iter + 1) % 1000 == 0:
        #     print('validating.............................................................')
        #     miou2 = val_segformer_IJCAI(model,n_iter + 1,'./0301.txt')
        #     if miou2 >= miou:
        #         torch.save(model.state_dict(),
        #                    "./checkpoint/model" + "_medical_fusion_best.pth")
        #         miou = miou2
        #         flag = True
        #     print('.............................................................')
        if (n_iter + 1) % 500 == 0:
            print('validating.............................................................')
            miou2 = val_segformer_IJCAI(model,n_iter + 1,'./0301.txt')
            if miou2 >= miou:
                torch.save(model.cpu().state_dict(),
                        "./checkpoint/model" + "_testab1_2mri_BI3.pth")
                model.to(device)  # 保存完再搬回 GPU
                miou = miou2
                flag = True
            print('.............................................................')


    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.local_rank == 0:
        setup_logger()
        logging.info('\nconfigs: %s' % cfg)

    train(cfg=cfg)
