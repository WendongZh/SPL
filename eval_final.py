from __future__ import print_function
import argparse
from math import log10
import numpy as np
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_test import build_dataloader
import pdb
import socket
import time
import skimage
#from skimage.measure import compare_ssim
#from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import io
from models_inpaint import InpaintingModel



# Testing settings
parser = argparse.ArgumentParser(description='SPL_test')
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--dataset', type=str, default='paris', help='used dataset, paris, places2, celeba')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--cpu', default=False, action='store_true', help='Use CPU to test')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--img_flist', type=str, default='/data/dataset/places2/flist/val.flist')
parser.add_argument('--mask_flist', type=str, default='/data/dataset/places2/flist/3w_all.flist')
parser.add_argument('--mask_index', type=str, default='selected_mask_fortest')
parser.add_argument('--model', default='./checkpoints', help='sr pretrained base model')
parser.add_argument('--save', default=False, action='store_true', help='If save test images')
parser.add_argument('--save_path', type=str, default='./test_results_place2')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=0.1)


opt = parser.parse_args()


def eval(device):
    model.eval()
    count_batch = 1
    avg_psnr, avg_ssim, avg_l1 = 0., 0., 0.
    for batch in testing_data_loader:
        gt_512_batch, gt_batch, mask_batch, index, name = batch
        t_io2 = time.time()
        if cuda:
            gt_batch = gt_batch.cuda(device)
            gt_512_batch = gt_512_batch.cuda(device)
            mask_batch = mask_batch.cuda(device)
            mask_batch = torch.mean(mask_batch, 1, keepdim=True)

        with torch.no_grad():
            mask_512 = F.interpolate(mask_batch, 512)
            gt_256_masked = gt_batch * (1.0 - mask_batch) + mask_batch
            gt_512_masked = F.interpolate(gt_256_masked, 512)

            prediction, _ = model.generator(gt_batch, mask_batch, gt_512_masked, mask_512)
            prediction = prediction * mask_batch + gt_batch * (1 - mask_batch)

        batch_avg_psnr, batch_avg_ssim, batch_avg_l1 = evaluate_batch(
            batch_size=opt.bs,
            gt_batch=gt_batch,
            pred_batch=prediction,
            mask_batch=mask_batch,
            save=opt.save,
            path=opt.save_path,
            count=count_batch,
            # index=index
            name=name
            )

        # avg_psnr = (avg_psnr * (count - 1) + batch_avg_psnr) / count
        avg_psnr = avg_psnr + ((batch_avg_psnr- avg_psnr) / count_batch)
        avg_ssim = avg_ssim + ((batch_avg_ssim- avg_ssim) / count_batch)
        avg_l1 = avg_l1 + ((batch_avg_l1- avg_l1) / count_batch)

        print(
            "Number: %05d" % (count_batch * opt.bs),
            " | Average: PSNR: %.4f" % (avg_psnr),
            " SSIM: %.4f" % (avg_ssim),
            " L1: %.4f" % (avg_l1),
            "| Current batch:", count_batch,
            " PSNR: %.4f" % (batch_avg_psnr),
            " SSIM: %.4f" % (batch_avg_ssim),
            " L1: %.4f" % (batch_avg_l1), flush=True
        )

        count_batch+=1




def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    io.imsave(path+'/'+name+'.png', img)

def PSNR(pred, gt, shave_border=0):
    return compare_psnr(pred, gt, data_range=255)
    # imdff = pred - gt
    # rmse = math.sqrt(np.mean(imdff ** 2))
    # if rmse == 0:
    #     return 100
    # return 20 * math.log10(255.0 / rmse)

def L1(pred, gt):
    return np.mean(np.abs((np.mean(pred,2) - np.mean(gt,2))/255))

def SSIM(pred, gt, data_range=255, win_size=11, multichannel=True):
    return compare_ssim(pred, gt, data_range=data_range, \
    multichannel=multichannel, win_size=win_size)

def evaluate_batch(batch_size, gt_batch, pred_batch, mask_batch, save=False, path=None, count=None, index=None, name=None):
    pred_batch = pred_batch * mask_batch + gt_batch * (1 - mask_batch)

    if save:
        input_batch = gt_batch * (1 - mask_batch) + mask_batch
        input_batch = (input_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
        mask_batch = (mask_batch.detach().permute(0,2,3,1).cpu().numpy()[:,:,:,0]*255).astype(np.uint8)

        if not os.path.exists(path):
            os.mkdir(path)


    gt_batch = (gt_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
    pred_batch = (pred_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)

    psnr, ssim, l1 = 0., 0., 0.
    for i in range(batch_size):
        if index == None:
            gt, pred = gt_batch[i], pred_batch[i]
        else:
            gt, pred, name = gt_batch[i], pred_batch[i], index[i].data.item()

        psnr += PSNR(pred, gt)
        ssim += SSIM(pred, gt)
        l1 += L1(pred, gt)

        if save:
            #save_img(path, str(count)+'_'+str(name)+'_input', input_batch[i])
            #save_img(path, str(count)+'_'+str(name)+'_mask', mask_batch[i])
            save_img(path, 'pred_'+name[i], pred_batch[i])
            #save_img(path, str(count)+'_'+str(name)+'_gt', gt_batch[i])

    return psnr/batch_size, ssim/batch_size, l1/batch_size



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':
    if opt.cpu:
        print("===== Use CPU to Test! =====")
    else:
        print("===== Use GPU to Test! =====")

    ## Set the GPU mode
    gpus_list=[0] #range(opt.gpus)
    cuda = not opt.cpu
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    # Model
    model = InpaintingModel(g_lr=opt.lr, d_lr=(opt.lr), l1_weight=opt.l1_weight, gan_weight=opt.gan_weight, iter=0, threshold=opt.threshold)

    pretrained_model = torch.load(opt.model, map_location=lambda storage, loc: storage)



    if cuda:
        device = torch.device('cuda:0')
        model = model.cuda(device)
        if len(gpus_list) > 1:
            model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
            model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpus_list)
            model.load_state_dict(pretained_model)
        else:
            state_dict = model.state_dict()
            new_dict_no_module = {}
            for k, v in pretrained_model.items():
                k = k.replace('module.', '')
                new_dict_no_module[k] = v

            new_dict = {k: v for k, v in new_dict_no_module.items() if k in state_dict.keys()}
            state_dict.update(new_dict)
            model.load_state_dict(state_dict)
        
    print('Pre-trained G model is loaded.')

    # Datasets
    print('===> Loading datasets')
    testing_data_loader = build_dataloader(
        dataset_name=opt.dataset,
        flist=opt.img_flist,
        mask_flist=opt.mask_flist,
        test_mask_index=opt.mask_index,
        augment=False,
        training=False,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=False
    )
    print('===> Loaded datasets')

    ## Eval Start!!!!
    eval(device)
