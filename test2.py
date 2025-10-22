# coding:utf-8
import os
import argparse
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model_TII import BiSeNet
from TaskFusion_dataset2 import Fusion_dataset
# from FusionNet import FusionNet
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import os, argparse, time, datetime, sys, shutil, stat
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat
# from core.model_fusion import Network, Network3, Fusion_Network3
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
parser.add_argument('--batch_size', '-B', type=int, default=1)
parser.add_argument('--gpu', '-G', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py

def main():
    conf_total = np.zeros((2, 2))
    fusion_model_path ='./checkpoint/model_testab1_2mri_BI3.pth'
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)
    model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab/'
    test_dataset = Fusion_dataset('val', ir_path=vi_path, vi_path=ir_path, label_path= label_path)
    print(fusion_model_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
            logits,_, seg1, = model.forward(images_ir,images_vis)
            images_vis_ycrcb = images_vis
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # fusion_gray = torch.cat(
            # (logits, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
            #                             dim=1,  # 在通道维度拼接
            #                                 )
            fusion_image = logits  # 保持灰度图，不需要转换到RGB
            #图像裁剪与保存
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = np.uint8(255.0 * fused_image)
            #保存融合后的图像
            fused_image = fused_image.transpose((0, 2, 3, 1))
            #归一化
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
            print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  # 分类2种
            conf_total += conf
            # save demo images
            visualize(image_name=name, predictions=seg1.argmax(1), weight_name='0')

        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        # conf_total_matfile = os.path.join("./runs", 'conf_' + args.weight_name + '.mat')
        # savemat(conf_total_matfile, {'conf': conf_total})  # 'conf' is the variable name when loaded in Matlab

        print('\n###########################################################################')
        # print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
        # args.model_name, args.weight_name, 1, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
        # print('\n* the tested dataset name: %s' % args.dataset_split)
        # print('* the tested image count: %d' % len(test_loader))
        # print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
        # print('* the weight name: %s' % args.weight_name)
        # print('* the file name: %s' % args.file_name)
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1])
        )
        # 打印 iou_per_class 的结果
        print(
            "*iou_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (iou_per_class[0], iou_per_class[1])
        )
        # 打印平均值
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        # print(
        #     '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (
        #     batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
        #     1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
        # print('\n* the total confusion matrix: ')
        # np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)
        # print(conf_total)
        print('\n###########################################################################')

def val_segformer_fused(model,epoch,file,method):
    file = file
    file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8#这一步的作用是什么
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    #vi_path = '/root/data1/cvpr_seg/train_all/'+method+'/'
    #ir_path ='/root/data1/cvpr_seg/train_all/Infrared/'
    #label_path='/root/data1/cvpr_seg/train_all/Label/'
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    file_method = './our.txt'
    file_method = open(file_method,'w+')

    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
            seg1 = model.forward(images_vis)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  
            precision_per_class, recall_per_class, iou_per_class = compute_results(conf)
            
            file_method.write(
            "%.6f" \
            % (
            iou_per_class[1]))
            # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1])
        )
        # 打印 iou_per_class 的结果
        print(
            "*iou_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (iou_per_class[0], iou_per_class[1])
        )
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        #
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1], iou_per_class[2]))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()


def val_segformer_IJCAI(model,epoch,file):
    file = file
    file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model.eval().cuda()
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')

    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.cuda()
                images_ir = images_ir.cuda()

            _, seg1 = model.forward(images_ir,images_vis)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
        
        if precision_per_class is None or iou_per_class is None:
            print("Error: compute_results returned None. Exiting.")
            return None
        print('\n###########################################################################')

        # 打印 precision_per_class 的结果
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1])
        )
        # 打印 iou_per_class 的结果
        print(
            "*iou_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (iou_per_class[0], iou_per_class[1])
        )
        # 打印平均值
        print("\n*average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
            % (precision_per_class.mean(), iou_per_class.mean()))
        print("*average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
            % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        print('\n###########################################################################')
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()

def val_segformer_fused2(model,epoch,file,method):
    file = file
    file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis).to(device)
            images_ir = Variable(images_ir).to(device)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
            seg1 = model.forward(torch.cat([images_ir,images_ir,images_ir],dim=1))
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf


        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        print(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()

def val_segformer(model,epoch):
    file = '/root/val_seg_add.txt'
    file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)

            logits,_, seg1, = model.forward(images_vis)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf


        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        print(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write('\n###########################################################################'+'\n')
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        file_o.write("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        file_o.write("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()

def val_segformer2(model,epoch,strategy):
    file = './val_seg_'+strategy+'.txt'
    file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)

            logits,_, seg1, = model.forward(images_vis)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)

            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf


        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        print(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write('\n###########################################################################'+'\n')
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        file_o.write("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        file_o.write("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()

def val_fusion(model,model2,epoch):

    conf_total = np.zeros((2, 2))

    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8

    device = torch.device("cuda:0")

    model.eval().to(device)
    model2.eval().to(device)

    # if args.gpu >= 0:
    #     model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
                ## RGB to tensor
                image_mask = np.array(Image.open(mask_path + name[0]))[:,:,np.newaxis]
                image_mask = np.concatenate([image_mask,image_mask,image_mask],axis=2)
                image_mask = (
                        np.asarray(Image.fromarray(image_mask), dtype=np.float32).transpose(
                            (2, 0, 1)
                        )
                        / 255.0
                )
                image_mask = np.expand_dims(image_mask, axis=0)
                image_mask = torch.tensor(image_mask).cuda()
                out0, out1 = model2.denoise_net.encoder.forward_fusion(image_mask)
                image_fusion = model(images_ir, images_vis,out0,out1)

                images_vis_ycrcb = images_vis
                fusion_gray = torch.cat(
                (image_fusion, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
                fusion_image = fusion_gray  # 保持灰度图，不需要转换到RGB 
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = np.uint8(255.0 * fused_image)

                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )

                fused_image = np.uint8(255.0 * fused_image)
                for k in range(len(name)):
                    image = fused_image[k, :, :, :]
                    image = Image.fromarray(image)
                    save_path = os.path.join('/root/data1/cvpr_seg/train_all/Mask/', name[k])
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))

def val_fusion_strategy(model,model2,epoch,strategy):

    conf_total = np.zeros((9, 9))

    n_classes = 9
    score_thres = 0.7
    ignore_idx = 255
    h=480
    w=640

    image_size=(h,w)
    n_min = 8 * 640 * 480 // 8

    device = torch.device("cuda:0")

    model.eval().to(device)
    model2.eval().to(device)
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
                ## RGB to tensor
                image_mask = np.array(Image.open(mask_path + name[0]))[:,:,np.newaxis]
                image_mask = np.concatenate([image_mask,image_mask,image_mask],axis=2)
                image_mask = (
                        np.asarray(Image.fromarray(image_mask), dtype=np.float32).transpose(
                            (2, 0, 1)
                        )
                        / 255.0
                )
                image_mask = np.expand_dims(image_mask, axis=0)
                image_mask = torch.tensor(image_mask).cuda()
                out0, out1 = model2.denoise_net.encoder.forward_fusion(image_mask)
                image_fusion = model(images_ir, images_vis,out0,out1)

                images_vis_ycrcb = images_vis
                fusion_gray = torch.cat(
                 (image_fusion, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
                fusion_image = fusion_gray 
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = np.uint8(255.0 * fused_image)

                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )

                fused_image = np.uint8(255.0 * fused_image)
                for k in range(len(name)):
                    image = fused_image[k, :, :, :]
                    image = Image.fromarray(image)
                    save_path = os.path.join('/root/data/'+strategy+'/', name[k])
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))

def val_fusion_write(epoch):

    conf_total = np.zeros((2, 2))

    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8

    device = torch.device("cuda:0")
    model2 = Network3(cfg.exp.backbone,cfg.dataset.num_classes).cuda()
    model2.load_state_dict(torch.load('./5619/model-fusion_add_final.pth'))

    model = Fusion_Network3()

    model.load_state_dict(torch.load('./5619/modelfusion-final.pth'))

    model.eval().to(device)
    model2.eval().to(device)

    # if args.gpu >= 0:
    #     model.eval().to(device)
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path= label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
                ## RGB to tensor
                image_mask = np.array(Image.open(mask_path + name[0]))[:, :, np.newaxis]
                image_mask = np.concatenate([image_mask, image_mask, image_mask], axis=2)
                image_mask = (
                        np.asarray(Image.fromarray(image_mask), dtype=np.float32).transpose(
                            (2, 0, 1)
                        )
                        / 255.0
                )
                image_mask = np.expand_dims(image_mask, axis=0)
                image_mask = torch.tensor(image_mask).cuda()
                out0, out1 = model2.denoise_net.encoder.forward_fusion(image_mask)
                image_fusion = model(images_ir, images_vis,out0,out1)

                images_vis_ycrcb = images_vis
                fusion_gray = torch.cat(
                (image_fusion, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
                fusion_image = fusion_gray 
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = np.uint8(255.0 * fused_image)

                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )

                fused_image = np.uint8(255.0 * fused_image)
                for k in range(len(name)):
                    image = fused_image[k, :, :, :]
                    image = Image.fromarray(image)
                    save_path = os.path.join('./test/', name[k])
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))
def val_fusion_train(model,model2, epoch):
    conf_total = np.zeros((2, 2))

    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h = 256
    w = 256

    image_size = (h, w)
    n_min = 8 * 256 * 256 // 8

    device = torch.device("cuda:0")

    model.eval().to(device)
    model2.eval().to(device)

    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'

    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path=label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, label, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
                ## RGB to tensor
                image_mask = np.array(Image.open(mask_path + name[0]))[:, :, np.newaxis]
                image_mask = np.concatenate([image_mask, image_mask, image_mask], axis=2)
                image_mask = (
                        np.asarray(Image.fromarray(image_mask), dtype=np.float32).transpose(
                            (2, 0, 1)
                        )
                        / 255.0
                )
                image_mask = np.expand_dims(image_mask, axis=0)
                image_mask = torch.tensor(image_mask).cuda()
                out0, out1 = model2.denoise_net.encoder.forward_fusion(image_mask)
                image_fusion = model(images_ir, images_vis,out0,out1)
                images_vis_ycrcb = images_vis
                fusion_gray = torch.cat(
                 (image_fusion, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
                fusion_image = fusion_gray  # 保持灰度图，不需要转换到RGB
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = np.uint8(255.0 * fused_image)

                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )

                fused_image = np.uint8(255.0 * fused_image)
                for k in range(len(name)):
                    image = fused_image[k, :, :, :]
                    image = Image.fromarray(image)
                    save_path = os.path.join('/root/data1/cvpr_seg/train_all/Mask/', name[k])
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))

def val_fusion_train_strategy(model,model2, epoch,strategy):
    conf_total = np.zeros((2, 2))

    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h = 256
    w = 256

    image_size = (h, w)
    n_min = 8 * 256 * 256 // 8

    device = torch.device("cuda:0")

    model.eval().to(device)
    model2.eval().to(device)

    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'

    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path=label_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, label, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
                ## RGB to tensor
                image_mask = np.array(Image.open(mask_path + name[0]))[:, :, np.newaxis]
                image_mask = np.concatenate([image_mask, image_mask, image_mask], axis=2)
                image_mask = (
                        np.asarray(Image.fromarray(image_mask), dtype=np.float32).transpose(
                            (2, 0, 1)
                        )
                        / 255.0
                )
                image_mask = np.expand_dims(image_mask, axis=0)
                image_mask = torch.tensor(image_mask).cuda()
                out0, out1 = model2.denoise_net.encoder.forward_fusion(image_mask)
                image_fusion = model(images_ir, images_vis,out0,out1)
                images_vis_ycrcb = images_vis
                fusion_gray = torch.cat(
                 (image_fusion, images_vis[:, 0:1, :, :]),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
                fusion_image = fusion_gray  # 保持灰度图，不需要转换到RGB)
                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                fused_image = fusion_image.cpu().numpy()
                fused_image = np.uint8(255.0 * fused_image)

                fused_image = fused_image.transpose((0, 2, 3, 1))
                fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )

                fused_image = np.uint8(255.0 * fused_image)
                for k in range(len(name)):
                    image = fused_image[k, :, :, :]
                    image = Image.fromarray(image)
                    save_path = os.path.join('/root/data1/cvpr_seg/train_all/'+strategy+'/', name[k])
                    image.save(save_path)
                    print('Fusion {0} Sucessfully!'.format(save_path))

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def val_segformer_IJCAI2(model,epoch):
    # file = file
    # file_o = open(file,'a+')
    conf_total = np.zeros((2, 2))
    # fusion_model_path ='./checkpoint/model-fusion_meta_55000.pth'
    # fusionmodel = eval(search-EXsearch-EXP-20220319-204533/model/179.ptP-20220319-221656/model/39.pt'FusionNet')(output=1)
    n_classes = 2
    score_thres = 0.7
    ignore_idx = 255
    h=256
    w=256

    image_size=(h,w)
    n_min = 8 * 256 * 256 // 8
    # model = Network(None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True)
    # model = Network(None, None, image_size, n_classes)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model.eval().cuda()
    # model.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    vi_path = './CT_image/'
    ir_path ='./MRI_image/'
    label_path='./Seglab'
    mask_path = '/home/yangxiangjun/BDL_Seg/MRI_image/'
   
    
    test_dataset = Fusion_dataset('val', ir_path=vi_path, vi_path=ir_path, label_path= label_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    file_method = './ours.txt'
    file_method = open(file_method,'w+')
    with torch.no_grad():
        for it, (images_vis, images_ir,label,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.cuda()
                images_ir = images_ir.cuda()

            image_fusion, seg1 = model.forward(images_ir,images_vis)
            seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)
            images_vis_ycrcb = images_vis
            fusion_gray = torch.cat(
            (image_fusion, image_fusion,image_fusion),  # 拼接融合图像与原始灰度图像的单通道部分
                                        dim=1,  # 在通道维度拼接
                                            )
            fusion_image = fusion_gray  # 保持灰度图，不需要转换到RGB)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            # fused_image = np.uint8(255.0 * fused_image)

            fused_image = fused_image.transpose((0, 2, 3, 1))
            # fused_image = (fused_image - np.min(fused_image)) / (
            #         np.max(fused_image) - np.min(fused_image)
            # )

            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                print(np.shape(image))
                image = Image.fromarray(image)
                save_path = os.path.join('./FusionRes/', name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
            # print(np.shape(seg1))
            label = label.cpu().numpy().squeeze().flatten()
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1]) 
            
            precision_per_class, recall_per_class, iou_per_class = compute_results(conf)
            
            file_method.write(
            "%.6f\n" \
            % (
            iou_per_class[1]))
             # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            visualize(pth='./SegRes/',image_name=name, predictions=seg1.argmax(1), weight_name='0')

        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (precision_per_class[0], precision_per_class[1]))
        print(
            "* iou per class: \n    unlabeled: %.6f, tumor: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1]))
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))

        print('\n###########################################################################')
        return iou_per_class.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parser.parse_args()
    n_class = 2
    # seg_model_path = './model/Fusion/model_final.pth'
    # fusion_model_path = './model/Fusion/fusionmodel_final.pth'
    fused_dir = './Fusion_results_0228'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    from core.model_fusion import Network_IJCAI as Network
    model = Network(None,None,'mit_b1', 2, 256, True).cuda()
    # model.load_state_dict(torch.load("./checkpoint/model_medical_fusion_best_gesenet10000.pth"),strict=True)
    state_dict = torch.load("./checkpoint/model_testab1_2mri_BI3.pth")
    model.load_state_dict(state_dict, strict=False)

    val_segformer_IJCAI2(model,0)