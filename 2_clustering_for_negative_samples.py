from tqdm import tqdm
from torchvision import transforms
from tool.GenDataset import Stage1_InferDataset
import os.path
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import importlib
from torch.backends import cudnn
cudnn.enabled = True
import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
import models.resnet38d
import shutil
import random
import argparse

def train_split(input_image_path,  img_train_dir,  train_rate):
    image_list = os.listdir(input_image_path)
    image_number = len(image_list)
    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number)
    for file_name in train_sample:
        img_name = input_image_path +'/'+ file_name 
        img_dst = img_train_dir +'/'+ file_name 
        shutil.copyfile(img_name, img_dst)

def train_split_number(input_image_path,  img_train_dir,  train_number):
    image_list = os.listdir(input_image_path)
    train_number = train_number
    train_sample = random.sample(image_list, train_number)
    for file_name in train_sample:
        img_name = input_image_path +'/'+ file_name 
        img_dst = img_train_dir +'/'+ file_name 
        shutil.copyfile(img_name, img_dst)

def copy_folder(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_folder(s, d)
        else:
            shutil.copy2(s, d)




def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--cluster_k", default=6, type=int)
    args = parser.parse_args()
    datatags = ['train', 'val']
    for args.datatag in datatags:
        num_clusters=args.cluster_k
        if args.datatag == 'train':
            rate = 4
        else:
            rate = 10
        tmppath = 'DATA/MIDOG2021/tmp/'+args.datatag+'/'+str(num_clusters)+'_kmeans/'
        root_n = 'DATA/MIDOG2021/dataset_step1/'+args.datatag+'/n/'
        root_p = 'DATA/MIDOG2021/dataset_step1/'+args.datatag+'/p/'
        save_train_path = 'DATA/MIDOG2021/dataset_step1/'+args.datatag+'/n_balance/'
        num_positive = len(os.listdir(root_p))
        n_class = 2
        num_workers = 10
        batch_size = 20
        data_size, dims = batch_size, 1
        weights = 'init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params'
        network = "models.resnet38_cls"

        os.makedirs(save_train_path,exist_ok=True)

        if os.path.exists(tmppath):
            shutil.rmtree(tmppath)
        os.makedirs(tmppath)
        os.makedirs(save_train_path, exist_ok=True)

        model = getattr(importlib.import_module(network), 'Net')()
        weights_dict = models.resnet38d.convert_mxnet_to_torch(weights)
        model.load_state_dict(weights_dict, strict=False)
        model.eval()
        model.cuda()

        transform = transforms.Compose([transforms.ToTensor()])
        infer_dataset = Stage1_InferDataset(root_n, transform=transform)
        train_size = int(0.2 * len(infer_dataset))
        test_size = len(infer_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(infer_dataset, [train_size, test_size])
        train_data_loader=DataLoader(train_dataset,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    batch_size=batch_size,
                                    )
        infer_data_loader = DataLoader(infer_dataset,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    batch_size=batch_size,
                                    drop_last=True
                                    )

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        all_feature_fit = torch.zeros((0,4096))

        """获取特征"""
        #训练Kmeans
        for iter, (img_names, img_list) in enumerate(tqdm(train_data_loader)):
            feature= model.extract_feature(img_list.cuda())
            all_feature_fit = torch.cat((all_feature_fit,feature.detach().cpu()),0)
        cluster_ids_x, cluster_centers = kmeans(
        X=all_feature_fit, num_clusters=num_clusters, distance='euclidean'
        )

        print("-------------infer--------------")
        for i in range(num_clusters):
            filepath=tmppath+'/'+str(i)
            if not os.path.exists(filepath):
                os.mkdir(filepath)

        for iter, (img_names, img_list) in enumerate(tqdm(infer_data_loader)):
            feature= model.extract_feature(img_list.cuda())
            cluster_ids_y = kmeans_predict(
                feature, cluster_centers, 'euclidean', device=device
            )
            id=np.array(cluster_ids_y.detach().cpu())
            for i in range(len(img_names)):

                src=root_n+'/'+img_names[i]+'.png'
                dst=tmppath+'/'+str(id[i])+'/'+img_names[i]+'.png'
                shutil.copy2(src=src,dst=dst)
        basename = tmppath
        for i in tqdm(range(num_clusters)):      
            
            input_image_path=basename+'/'+str(i)
            train_split_number(input_image_path,  save_train_path,  num_positive)

if __name__ == '__main__':
    main()
    shutil.rmtree('DATA/MIDOG2021/dataset_step1/val/n/')