import glob
import os, utils
from argparse import ArgumentParser
import torch.utils.data as Data
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path
import pystrum.pynd.ndutils as nd

from Model_corr_and_fourier import TeP
from data import datasets, trans

import time
from torchinfo import summary
from medpy.metric.binary import hd,hd95,dc,assd

import pickle as pkl
from tqdm import tqdm



parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str,
                    dest="dataset_name",
                    default='OASIS')

parser.add_argument("--imgshape", type=str,
                    dest="imgshape",
                    default='(128,128,128)')

parser.add_argument("--test_dir", type=str,
                    dest="test_dir",
                    default='E:/docs/under_graduate/datasets/neurite_oasis_preprocessed/Test/')
parser.add_argument("--model_dir", type=str,
                    dest="model_dir",
                    default='./Experiments_OASIS/model_corr_and_fourier/')
parser.add_argument("--model_file", type=str,
                    dest="model_file",
                    default='TeP_correlation_0.81016_iter156480.pth')
opt = parser.parse_args()

dataset_name=opt.dataset_name
imgshape=eval(opt.imgshape)
test_dir=opt.test_dir
model_dir=opt.model_dir
model_file=opt.model_file
model_path=os.path.join(model_dir,model_file)
assert dataset_name in ['OASIS', 'LPBA', 'Mindboggle'], 'dataset name {} invalid'.format(dataset_name)
if dataset_name == 'OASIS':
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 35]
elif dataset_name == 'LPBA':
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                53, 54]  # 54
elif dataset_name == 'Mindboggle':
    VOI_lbls = [i + 1 for i in range(40)]  # 54

def plot_brain_and_save(input_fixed_arr, moving_image, warped, dir_path, case_idx, mode="eval"):
    # visualize
    # sagittal
    input_fixed_arr = input_fixed_arr.squeeze()
    moving_image = moving_image.squeeze()
    warped = warped.squeeze()

    x,y,z = input_fixed_arr.shape

    name_list=['input_fixed_arr','moving_image','warped']
    fig = plt.figure(figsize=(9, 9))
    idx=1
    for i,name in enumerate(name_list):
        img=eval(name)

        ax = fig.add_subplot(330+idx)
        ax.axis('off')
        plt.tight_layout()
        ax.imshow(img[:, :, z//2], cmap='Greys')
        idx=idx+1

        ax = fig.add_subplot(330 + idx)
        ax.axis('off')
        plt.tight_layout()
        ax.imshow(img[:, y//2, :], cmap='Greys')
        idx = idx + 1

        ax = fig.add_subplot(330 + idx)
        ax.axis('off')
        plt.tight_layout()
        ax.imshow(img[x//2, :, :], cmap='Greys')
        idx = idx + 1

    # save fig
    fig.savefig(os.path.join(dir_path, 'case_{}.jpg'.format(case_idx)))

    plt.close(fig)

def dice_hd_assd(pred1, truth1, VOI_lbls):

    dices = np.zeros(len(VOI_lbls))
    hds = np.zeros(len(VOI_lbls))
    assds = np.zeros(len(VOI_lbls))
    index = 0
    for k in tqdm(VOI_lbls):
        # print(k)
        truth = truth1 == k
        pred = pred1 == k

        assert truth.any() == True and pred.any() == True, 'empty VOI_lbl: {}'.format(k)

        intersection = np.sum(pred * truth) * 2.0
        dices[index] = intersection / (np.sum(pred) + np.sum(truth))
        hds[index] = hd95(pred, truth)
        assds[index] = assd(pred, truth)

        index = index + 1
    return dices, hds, assds

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def main():

    reg_model = utils.register_model(imgshape, 'nearest')

    Quantitative_Results_dir=os.path.join(model_dir,'Quantitative_Results')
    Visualize_Results_dir=os.path.join(model_dir,'Visualize_Results')
    if not os.path.exists(Quantitative_Results_dir):
        os.makedirs(Quantitative_Results_dir)
    if not os.path.exists(Visualize_Results_dir):
        os.makedirs(Visualize_Results_dir)

    model = TeP(start_channel=16, inshape=imgshape)
    best_model = torch.load(model_path,map_location='cpu')['model_state']
    model.load_state_dict(best_model)

    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.to(device)
    test_composed = transforms.Compose(
        [
            trans.NumpyType((np.float32, np.int16))
        ]
    )

    test_set = datasets.OASIS_InferDataset(root_dir=test_dir, transforms=test_composed)

    test_loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                  drop_last=True)

    infer_time = utils.AverageMeter()
    jac = utils.AverageMeter()

    dice = utils.AverageMeter()
    dice_details = np.zeros((test_set.__len__(), len(VOI_lbls)))
    hd95 = utils.AverageMeter()
    hd95_details = np.zeros((test_set.__len__(), len(VOI_lbls)))
    assd = utils.AverageMeter()
    assd_details = np.zeros((test_set.__len__(), len(VOI_lbls)))

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            print('case :', stdy_idx, '................')
            model.eval()
            data = [t for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            start = time.time()
            warped,flow,_ = model(x.float(), y.float())
            end = time.time()
            infer_time.update(end - start)
            warped_xv_seg = reg_model([x_seg.float(), flow])
            dices, hd95s, assds = dice_hd_assd(
                warped_xv_seg[0, ...].data.cpu().numpy().copy(),
                y_seg[0, ...].data.cpu().numpy().copy(),
                VOI_lbls
            )
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            jac.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            dice_details[stdy_idx:stdy_idx + 1, :] = dices[None, :]
            dice.update(np.mean(dices))

            hd95_details[stdy_idx:stdy_idx + 1, :] = hd95s[None, :]
            hd95.update(np.mean(hd95s))

            assd_details[stdy_idx:stdy_idx + 1, :] = assds[None, :]
            assd.update(np.mean(assds))
            print('dice:', np.mean(dices), ' and hd95 mean is :', np.mean(hd95s), ' and assd is:',
                  np.mean(assds),'and jac is:',jac.val)

            '''plot_brain_and_save(input_fixed_arr=y.detach().cpu().numpy(),
                                moving_image=x.detach().cpu().numpy(),
                                warped=warped.detach().cpu().numpy(),
                                dir_path=Visualize_Results_dir,
                                case_idx=stdy_idx,
                                mode='test')'''

            '''print('GPU usage:{0},{1:.4f},{2},{3:.4f}'.format(torch.cuda.memory_allocated(device=device),
                                                torch.cuda.memory_allocated(device=device)/1024/1024,
                                                torch.cuda.memory_reserved(device),
                                               torch.cuda.memory_reserved(device)/1024/1024))'''

            stdy_idx += 1


        print('time:', infer_time.avg, '---std:', infer_time.std)
        print('mean jac:', jac.avg, '---std:', jac.std)
        print('mean DSC:', dice.avg, '---std:', dice.std)
        print('mean hd95:', hd95.avg, '---std:', hd95.std)
        print('mean assd:', assd.avg, '---std:', assd.std)
        print(

            round(dice.avg, 5),'±',round(dice.std, 3),
            round(hd95.avg, 3),'±',round(hd95.std, 3),
            round(assd.avg, 3),'±',round(assd.std, 3),
            round(infer_time.avg, 3), '±', round(infer_time.std, 3),
            round(jac.avg,6),'±',round(jac.std,6)
        )

        with open(os.path.join(Quantitative_Results_dir,'dice_details.pkl'),'wb') as f:
            pkl.dump(dice_details,f)
        with open(os.path.join(Quantitative_Results_dir,'hd95_details.pkl'),'wb') as f:
            pkl.dump(hd95_details,f)
        with open(os.path.join(Quantitative_Results_dir,'assd_details.pkl'),'wb') as f:
            pkl.dump(assd_details,f)

if __name__ == '__main__':

    main()