from tqdm import tqdm
from torchvision import transforms
from tool.GenDataset import Stage1_InferDataset
import cv2
import os.path
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from pdb import set_trace
import os
from cv2 import norm
import torch
import importlib
from torch.backends import cudnn
cudnn.enabled = True
import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cluster_k', default=2, type=int)
    parser.add_argument('--cluster_k_2', default=4, type=int)
    args = parser.parse_args()


    num_workers = 0
    cluster_k_2=args.cluster_k_2
    cluster_k=args.cluster_k
    if cluster_k_2 == 2:
        temp_p=['(1000)','(0100)']
        temp_n=['(0010)','(0001)']
    elif cluster_k_2 == 4:
        temp_p=['(10000000)','(01000000)','(00100000)','(00010000)']
        temp_n=['(00001000)','(00000100)','(00000010)','(00000001)']
    elif cluster_k_2 == 6:
        temp_p=['(100000000000)','(010000000000)','(001000000000)','(000100000000)', '000010000000', '000001000000']
        temp_n=['(000000100000)','(000000100000)','(000000001000)','(000000000100)', '000000000010', '000000000001']
    # if pos_or_neg == 'n':
    #     temp=temp_n
    # elif pos_or_neg == 'p':
    #     temp=temp_p
    temp=temp_p
    batch_size=300
    train_root = 'DATA/MIDOG2021/dataset_step1/train/n/'
    infer_root=train_root
    savepath = 'DATA/MIDOG2021_HN/train/90_n_v2'
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)
    data_size, dims, num_clusters = batch_size, 1, cluster_k_2
    network = "network.resnet38_cls_dassl_multi_label"

    # # weights = 'save_weights/stage1/k='+str(args.cluster_k)+'/best_model.pth'
    # weights = 'save_weights/reproduction_batch_size=400/k1=4_k2=4/best_model.pth'
    # model = getattr(importlib.import_module(network), 'Net')(2,8)
    # weights_dict = torch.load(weights)
    # model.load_state_dict(weights_dict, strict=True)
    import models.resnet38d as models
    model = getattr(importlib.import_module(network), 'Net')(2,4)
    weights_dict = models.convert_mxnet_to_torch('init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
    model.load_state_dict(weights_dict, strict=False)

    model.eval()
    model.cuda()

    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(infer_root, transform=transform)
    train_dataset=Stage1_InferDataset(train_root, transform=transform)
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
    X=all_feature_fit, num_clusters=num_clusters, distance='euclidean', device=device
    )
    feature_all=[]
    ids_all=[]
    for iter, (img_names, img_list) in enumerate(tqdm(infer_data_loader)):
        feature= model.extract_feature(img_list.cuda())
        cluster_ids_y = kmeans_predict(
            feature, cluster_centers, 'euclidean', device=device
        )
        id=np.array(cluster_ids_y.detach().cpu())
        feature=np.array(feature.detach().cpu())

        ids_all.append(id)
        feature_all.append(feature)

        for i in range(len(img_names)):
            src=infer_root+'/'+img_names[i]+'.png'
            dst=savepath+'/'+img_names[i]+temp[id[i]]+'.png'
            shutil.copy2(src=src,dst=dst)

if __name__ == '__main__':
    main()