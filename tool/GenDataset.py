# -*- coding: utf-8 -*-
import os
from random import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tool.custom_transforms as tr
import numpy as np
class Stage1_InferDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.object = self.path_label()

    def __getitem__(self, index):
        fn = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_list = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                path_list.append(image_path)
        return path_list


class Stage1_TrainDataset(Dataset):
    def __init__(self, p_data_path,n_data_path, transform=None, target_transform=None, dataset=None):
        self.p_data_path = p_data_path
        self.n_data_path=n_data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.targets=[]
        
        self.object,self.targets= self.path_label()
        temp1=self.object
        temp2=self.targets
        temp = list(zip(temp1, temp2))
        from random import shuffle
        shuffle(temp)
        self.object[:], self.targets[:] = zip(*temp) 
        

    def __getitem__(self, index):
        fn, label = self.object[index] #fn为图片路径，label为对应图片的标签
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_object = []
        label=[]
        num_p=0
        for root, dirname, filename in os.walk(self.p_data_path):
            for f in filename:
                num_p=num_p+1
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
                
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
        for root, dirname, filename in os.walk(self.n_data_path):
            if num_p!=0:
                n_filename = np.random.choice(filename,num_p, True)#在阴性样本中挑选与阳性样本相同的样本数
            else:
                n_filename = np.random.choice(filename,len(filename), True)#当sampling推理时
            for f in n_filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
            
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
               
        return path_object,label

class Stage1_TrainDataset_multi_label(Dataset):
    def __init__(self, args, p_data_path,n_data_path, transform=None, target_transform=None, dataset=None):
        self.p_data_path = p_data_path
        self.n_data_path=n_data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.targets=[]
        
        self.object,self.targets= self.path_label(args)
        temp1=self.object
        temp2=self.targets
        temp = list(zip(temp1, temp2))
        from random import shuffle
        shuffle(temp)
        self.object[:], self.targets[:] = zip(*temp) 
        

    def __getitem__(self, index):
        fn, label,multi_label = self.object[index] #fn为图片路径，label为对应图片的标签
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return fn.split('/')[-1][:-4], img, label,multi_label
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self, args):
        path_object = []
        label=[]
        num_p=0
        for root, dirname, filename in os.walk(self.p_data_path):
            for f in filename:
                num_p=num_p+1
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
                
                multi_label_str=fname.split(')')[-2].split('(')[-1]
                if args.cluster_k == 2:
                    # 2
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3])
                         ], dtype=int)
                elif args.cluster_k == 4:
                    # 4
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3]),
                         int(multi_label_str[4]), int(multi_label_str[5]), int(multi_label_str[6]), int(multi_label_str[7])
                         ], dtype=int)
                elif args.cluster_k == 6:
                    # 6
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3]), \
                         int(multi_label_str[4]), int(multi_label_str[5]), int(multi_label_str[6]), int(multi_label_str[7]), \
                          int(multi_label_str[8]), int(multi_label_str[9]), int(multi_label_str[10]), int(multi_label_str[11])
                         ], dtype=int)

                path_object.append((image_path, image_label, multi_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
        for root, dirname, filename in os.walk(self.n_data_path):
            n_filename = np.random.choice(filename,num_p, False)#在阴性样本中挑选与阳性样本相同的样本数
            for f in n_filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)

                multi_label_str=fname.split(')')[-2].split('(')[-1]

                if args.cluster_k == 2:
                    # 2
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3])
                         ], dtype=int)
                elif args.cluster_k == 4:
                    # 4
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3]),
                         int(multi_label_str[4]), int(multi_label_str[5]), int(multi_label_str[6]), int(multi_label_str[7])
                         ], dtype=int)
                elif args.cluster_k == 6:
                    # 6
                    multi_label = torch.tensor(
                        [int(multi_label_str[0]), int(multi_label_str[1]), int(multi_label_str[2]), int(multi_label_str[3]), \
                         int(multi_label_str[4]), int(multi_label_str[5]), int(multi_label_str[6]), int(multi_label_str[7]), \
                          int(multi_label_str[8]), int(multi_label_str[9]), int(multi_label_str[10]), int(multi_label_str[11])
                         ], dtype=int)
                
                path_object.append((image_path, image_label,multi_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
               
        return path_object,label
    

class Stage1_TrainDataset_1(Dataset):
    def __init__(self, p_data_path,n_data_path, transform=None, target_transform=None, dataset=None):
        self.p_data_path = p_data_path
        self.n_data_path=n_data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.targets=[]
        
        self.object,self.targets= self.path_label_1()
        temp1=self.object
        temp2=self.targets
        temp = list(zip(temp1, temp2))
        from random import shuffle
        shuffle(temp)
        self.object[:], self.targets[:] = zip(*temp) 
        

    def __getitem__(self, index):
        fn, label = self.object[index] #fn为图片路径，label为对应图片的标签
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label_1(self):
        path_object = []
        label=[]
        num_p=0
        for root, dirname, filename in os.walk(self.p_data_path):
            for f in filename:
                num_p=num_p+1
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
                
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
        for root, dirname, filename in os.walk(self.n_data_path):
            n_filename = np.random.choice(filename,5*num_p, False)#在阴性样本中挑选与阳性样本相同的样本数
            for f in n_filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
            
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
               
        return path_object,label

class Stage1_TrainDataset_2(Dataset):
    def __init__(self, p_data_path,n_data_path, transform=None, target_transform=None, dataset=None):
        self.p_data_path = p_data_path
        self.n_data_path=n_data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.targets=[]
        
        self.object,self.targets= self.path_label_2()
        temp1=self.object
        temp2=self.targets
        temp = list(zip(temp1, temp2))
        from random import shuffle
        shuffle(temp)
        self.object[:], self.targets[:] = zip(*temp) 
        

    def __getitem__(self, index):
        fn, label = self.object[index] #fn为图片路径，label为对应图片的标签
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label_2(self):
        path_object = []
        label=[]
        num_p=0
        for root, dirname, filename in os.walk(self.p_data_path):
            for f in filename:
                num_p=num_p+1
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
                
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
        for root, dirname, filename in os.walk(self.n_data_path):
            n_filename = np.random.choice(filename,num_p//5, False)#在阴性样本中挑选与阳性样本相同的样本数
            for f in n_filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png

                label_str = fname.split(']')[0].split('[')[-1]
                
                image_label = torch.tensor([int(label_str[0]),int(label_str[1])],dtype=int)
            
                path_object.append((image_path, image_label))
                if(image_label[0]==torch.tensor(0)):
                    label.append(0)
                else:
                    label.append(1)
               
        return path_object,label

class Stage2_Dataset(Dataset):
    def __init__(self, args, base_dir, split):

        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if self.split   == "train":
            self._image_dir     = os.path.join(self._base_dir, 'train/')
            self._cat_dir       = os.path.join(self._base_dir, 'train_PM/PM_bn7/')
            self._cat_dir_a     = os.path.join(self._base_dir, 'train_PM/PM_b5_2')
            self._cat_dir_b     = os.path.join(self._base_dir, 'train_PM/PM_b4_5')
        elif self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'val/mask/')
        elif self.split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'test/mask/')
        self.args = args
        self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
        self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
        self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]
        if self.split == "train":
            self.categories_a = [os.path.join(self._cat_dir_a, fn + '.png') for fn in self.filenames]
            self.categories_b = [os.path.join(self._cat_dir_b, fn + '.png') for fn in self.filenames]
        assert (len(self.images) == len(self.categories))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == "train": 
            _img, _target, _target_a, _target_b = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target, 'label_a': _target_a, 'label_b': _target_b}
        elif (self.split == 'val') or (self.split == 'test'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        if self.split == "train":
            return self.transform_tr_ab(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            return self.transform_val(sample), image_dir
    def _make_img_gt_point_pair(self, index):
        if self.split == "train": 
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            _target_a = Image.open(self.categories_a[index])
            _target_b = Image.open(self.categories_b[index])
            return _img,_target,_target_a,_target_b
        elif (self.split == 'val') or (self.split == 'test'):
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img,_target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_tr_ab(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_ab(),
            tr.RandomGaussianBlur_ab(),
            tr.Normalize_ab(),
            tr.ToTensor_ab()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return None

def make_data_loader(args, **kwargs):

    train_set   = Stage2_Dataset(args, base_dir=args.dataroot, split='train')
    val_set     = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
    test_set    = Stage2_Dataset(args, base_dir=args.dataroot, split='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
