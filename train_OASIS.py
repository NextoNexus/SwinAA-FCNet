# 导入相关的库
import glob
import math

import os, losses, utils
import sys
import json
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from Model_corr import TeP
import random
import time


def same_seeds(seed):
    
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


same_seeds(123)

def dice(pred1, truth1, dataset_name='OASIS'):
    VOI_lbls = []

    assert dataset_name in ['OASIS', 'LPBA', 'Mindboggle'], 'dataset name {} invalid'.format(dataset_name)

    if dataset_name == 'OASIS':
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 35]
    elif dataset_name == 'LPBA':
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                    53, 54]#54
    elif dataset_name == 'Mindboggle':
        VOI_lbls = [i+1 for i in range(50)]#54

    dices = np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0

        dices[index] = intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dices)

def main():
    dataset_name='OASIS'
    model_name='Model_corr'
    start_channel=16
    parallel_sizes=[1,3,5]
    batch_size = 1
    iteration=160000
    n_checkpoint=320
    lr = 0.0001
    weights = [1, 1]  # loss weights
    img_size = (128, 128, 128)

    '''
    Initialize model
    '''
    model = TeP(start_channel=start_channel, inshape=img_size).cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model_nearest = utils.register_model(img_size, 'nearest').cuda()

    '''
    Initialize training
    '''

    train_dir = '/home/zhn/dataset/neurite_oasis/Train'
    val_dir = '/home/zhn/dataset/neurite_oasis/Val'
    train_composed = transforms.Compose(
        [
            trans.RandomFlip(0),
            trans.NumpyType((np.float32, np.float32)),
        ]
    )

    val_composed = transforms.Compose(
        [
            trans.NumpyType((np.float32, np.int16))
        ]
    )
    train_set = datasets.OASISDataset(root_dir=train_dir, transforms=train_composed)
    val_set = datasets.OASIS_InferDataset(root_dir=val_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            drop_last=True)

    max_epoch = math.ceil(iteration/train_set.__len__())

    opt = {
        'dataset_name':dataset_name,
        'model_name':model_name,
        'start_channel': start_channel,
        'parallel_sizes': parallel_sizes,
        'batch_size': batch_size,
        'iteration': iteration,
        'n_checkpoint': n_checkpoint,
        'lr': lr,
        'weight': weights,
        'max_epoch': max_epoch,
        'img_size': img_size
    }

    # save_dir = 'RDP_OASIS_ncc_{}_reg_{}_lr_{}_{}/'.format(*weights, lr,time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    save_dir = 'experiments_{}/'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    model_save_dir = 'Experiments_{}/'.format(dataset_name) + save_dir
    log_save_dir = 'Logs_{}/'.format(dataset_name) + save_dir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    with open(log_save_dir + 'options.txt', 'w') as f:
        json.dump(opt, f, indent=2)
    


    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    # writer = SummaryWriter(log_dir='logs/'+save_dir)
    epoch=0
    step=1
    while step <= iteration:
        loss_all = utils.AverageMeter()
        for data in train_loader:
            model.train()
            #adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            output = model(x, y)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {}--> Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(epoch,
                                                                                               step,
                                                                                               iteration,
                                                                                               loss.item(),
                                                                                               loss_vals[0].item(),
                                                                                               loss_vals[1].item()))
            if (step % n_checkpoint == 0):
                '''
                Validation
                '''
                print('start evaluation--------------------->')
                idx = 1
                eval_dsc = utils.AverageMeter()
                with torch.no_grad():
                    for data in val_loader:
                        model.eval()
                        data = [t.cuda() for t in data]
                        x = data[0]
                        y = data[1]
                        x_seg = data[2]
                        y_seg = data[3]
                        output = model(x, y)
                        def_out = reg_model_nearest([x_seg.cuda().float(), output[1].cuda()])
                        dsc = dice(def_out.data.cpu().numpy().copy(),
                                   y_seg.data.cpu().numpy().copy(),
                                   dataset_name=dataset_name
                                   )
                        eval_dsc.update(dsc, x.size(0))
                        print('step:', step, 'case:', idx, 'dice:', dsc)
                        idx = idx + 1
                print('mean dice after this validation:', eval_dsc.avg)
                best_dsc = max(eval_dsc.avg, best_dsc)
                with open(os.path.join(log_save_dir, 'losses and dice' + ".txt"), "a") as f:
                    print('{}:{}'.format(step,eval_dsc.avg), file=f)
                save_checkpoint({
                    'step': step,
                    'model_state': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir=model_save_dir, filename='{}_{:.5f}_iter{}.pth'.format(model_name,eval_dsc.avg, step))
            step += 1

            if step > iteration:
                break
        print('one epoch pass')
        epoch=epoch+1
        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth', max_model_num=10):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*.pth'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*.pth'))


if __name__ == '__main__':
    '''

    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()