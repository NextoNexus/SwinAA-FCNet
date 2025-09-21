import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np

class LPBADataset(Dataset):
    def __init__(self, root_dir, case_dirs, transforms):
        self.root_dir = root_dir
        self.case_dirs = case_dirs
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir,self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir,self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]

        src, tgt = self.transforms([src,tgt])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)

        src,tgt=torch.from_numpy(src),torch.from_numpy(tgt)
        return src,tgt

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)

class OASISDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir,self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir,self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]

        src, tgt = self.transforms([src,tgt])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)

        src,tgt=torch.from_numpy(src),torch.from_numpy(tgt)
        return src,tgt

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)


class LPBA_InferDataset(Dataset):
    def __init__(self, root_dir, case_dirs, transforms,need_casename=False):
        self.root_dir = root_dir
        self.case_dirs = case_dirs
        self.transforms = transforms
        self.need_casename = need_casename

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):

        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir, self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir, self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(src_file,'label.pkl'),'rb') as f:
            src_seg=pkl.load(f)

        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)
        with open(os.path.join(tgt_file,'label.pkl'),'rb') as f:
            tgt_seg=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]
        src_seg, tgt_seg = src_seg[None, ...], tgt_seg[None, ...]
        src, src_seg = self.transforms([src, src_seg])
        tgt, tgt_seg = self.transforms([tgt, tgt_seg])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)
        src_seg = np.ascontiguousarray(src_seg)  # [Bsize,channelsHeight,,Width,Depth]
        tgt_seg = np.ascontiguousarray(tgt_seg)
        src, tgt, src_seg, tgt_seg = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(src_seg), torch.from_numpy(tgt_seg)
        if self.need_casename == False:
            return src, tgt, src_seg, tgt_seg
        else:
            return src, tgt, src_seg, tgt_seg, self.case_dirs[x_index], self.case_dirs[y_index]
        #return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)

class OASIS_InferDataset(Dataset):
    def __init__(self, root_dir, transforms,need_casename=False):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms
        self.need_casename = need_casename

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        '''x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s'''
        x_index=index
        y_index=index+len(self.case_dirs)//2

        src_file = os.path.join(self.root_dir, self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir, self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(src_file,'label.pkl'),'rb') as f:
            src_seg=pkl.load(f)

        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)
        with open(os.path.join(tgt_file,'label.pkl'),'rb') as f:
            tgt_seg=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]
        src_seg, tgt_seg = src_seg[None, ...], tgt_seg[None, ...]
        src, src_seg = self.transforms([src, src_seg])
        tgt, tgt_seg = self.transforms([tgt, tgt_seg])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)
        src_seg = np.ascontiguousarray(src_seg)  # [Bsize,channelsHeight,,Width,Depth]
        tgt_seg = np.ascontiguousarray(tgt_seg)
        src, tgt, src_seg, tgt_seg = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(src_seg), torch.from_numpy(tgt_seg)
        if self.need_casename == False:
            return src, tgt, src_seg, tgt_seg
        else:
            return src, tgt, src_seg, tgt_seg, self.case_dirs[x_index], self.case_dirs[y_index]

    def __len__(self):
        return len(self.case_dirs)//2

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        #print('load',os.path.split(path)[-1],'...')
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class AbdomenDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir,self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir,self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]

        src, tgt = self.transforms([src,tgt])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)

        src,tgt=torch.from_numpy(src),torch.from_numpy(tgt)
        return src,tgt

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)


class Abdomen_InferDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):

        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir, self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir, self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(src_file,'label.pkl'),'rb') as f:
            src_seg=pkl.load(f)

        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)
        with open(os.path.join(tgt_file,'label.pkl'),'rb') as f:
            tgt_seg=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]
        src_seg, tgt_seg = src_seg[None, ...], tgt_seg[None, ...]
        src, src_seg = self.transforms([src, src_seg])
        tgt, tgt_seg = self.transforms([tgt, tgt_seg])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)
        src_seg = np.ascontiguousarray(src_seg)  # [Bsize,channelsHeight,,Width,Depth]
        tgt_seg = np.ascontiguousarray(tgt_seg)
        src, tgt, src_seg, tgt_seg = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(src_seg), torch.from_numpy(tgt_seg)
        return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)

class MindboggleDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir,self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir,self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]

        src, tgt = self.transforms([src,tgt])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)

        src,tgt=torch.from_numpy(src),torch.from_numpy(tgt)
        return src,tgt

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)


class Mindboggle_InferDataset(Dataset):
    def __init__(self, root_dir, transforms, need_casename=False):
        self.root_dir = root_dir
        self.case_dirs = os.listdir(root_dir)
        self.transforms = transforms
        self.need_casename = need_casename

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):

        x_index = index // (len(self.case_dirs) - 1)
        s = index % (len(self.case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.root_dir, self.case_dirs[x_index])
        tgt_file = os.path.join(self.root_dir, self.case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(src_file,'label.pkl'),'rb') as f:
            src_seg=pkl.load(f)

        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)
        with open(os.path.join(tgt_file,'label.pkl'),'rb') as f:
            tgt_seg=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]
        src_seg, tgt_seg = src_seg[None, ...], tgt_seg[None, ...]
        src, src_seg = self.transforms([src, src_seg])
        tgt, tgt_seg = self.transforms([tgt, tgt_seg])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)
        src_seg = np.ascontiguousarray(src_seg)  # [Bsize,channelsHeight,,Width,Depth]
        tgt_seg = np.ascontiguousarray(tgt_seg)
        src, tgt, src_seg, tgt_seg = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(src_seg), torch.from_numpy(tgt_seg)
        if self.need_casename:
            return src, tgt, src_seg, tgt_seg, self.case_dirs[x_index], self.case_dirs[y_index]
        return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.case_dirs) * (len(self.case_dirs) - 1)

class AbdomenMRCTDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.mri_root_dir = os.path.join(root_dir, 'MRI')
        self.ct_root_dir = os.path.join(root_dir, 'CT')

        self.mri_case_dirs = os.listdir(self.mri_root_dir)
        self.ct_case_dirs = os.listdir(self.ct_root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.mri_case_dirs) - 1)
        s = index % (len(self.mri_case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.mri_root_dir, self.mri_case_dirs[x_index])
        tgt_file = os.path.join(self.ct_root_dir, self.ct_case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]

        src, tgt = self.transforms([src,tgt])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)

        src,tgt=torch.from_numpy(src),torch.from_numpy(tgt)
        return src,tgt

    def __len__(self):
        return len(self.mri_case_dirs) * (len(self.mri_case_dirs) - 1)


class AbdomenMRCT_InferDataset(Dataset):
    def __init__(self, root_dir, transforms, need_casename=False):
        self.root_dir = root_dir
        self.need_casename=need_casename
        self.mri_root_dir=os.path.join(root_dir,'MRI')
        self.ct_root_dir=os.path.join(root_dir, 'CT')

        self.mri_case_dirs = os.listdir(self.mri_root_dir)
        self.ct_case_dirs = os.listdir(self.ct_root_dir)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):

        x_index = index // (len(self.mri_case_dirs) - 1)
        s = index % (len(self.mri_case_dirs) - 1)
        y_index = s + 1 if s >= x_index else s

        src_file = os.path.join(self.mri_root_dir, self.mri_case_dirs[x_index])
        tgt_file = os.path.join(self.ct_root_dir, self.ct_case_dirs[y_index])

        with open(os.path.join(src_file,'vol.pkl'),'rb') as f:
            src=pkl.load(f)
        with open(os.path.join(src_file,'label.pkl'),'rb') as f:
            src_seg=pkl.load(f)

        with open(os.path.join(tgt_file,'vol.pkl'),'rb') as f:
            tgt=pkl.load(f)
        with open(os.path.join(tgt_file,'label.pkl'),'rb') as f:
            tgt_seg=pkl.load(f)

        src, tgt = src[None, ...], tgt[None, ...]
        src_seg, tgt_seg = src_seg[None, ...], tgt_seg[None, ...]
        src, src_seg = self.transforms([src, src_seg])
        tgt, tgt_seg = self.transforms([tgt, tgt_seg])

        src = np.ascontiguousarray(src)  # [Bsize,channelsHeight,,Width,Depth]
        tgt = np.ascontiguousarray(tgt)
        src_seg = np.ascontiguousarray(src_seg)  # [Bsize,channelsHeight,,Width,Depth]
        tgt_seg = np.ascontiguousarray(tgt_seg)
        src, tgt, src_seg, tgt_seg = torch.from_numpy(src), torch.from_numpy(tgt), torch.from_numpy(src_seg), torch.from_numpy(tgt_seg)
        if self.need_casename:
            return src, tgt, src_seg, tgt_seg, self.mri_case_dirs[x_index], self.ct_case_dirs[y_index]
        return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.mri_case_dirs) * (len(self.mri_case_dirs) - 1)