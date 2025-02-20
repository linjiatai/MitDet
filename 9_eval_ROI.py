from PIL import Image
import os
import math
from tqdm import tqdm
import csv
import random
from numpy import *
import glob
import json
import pandas as pd
import os
import cv2
from tqdm import tqdm

cluster_k = 4
cluster_k_2 = 4

path="DATA/MIDOG2021/0_data_ori/val/"
# path = "/home/linjiatai/MitDet_linjiatai（最终）/DATA/MIDOG2022(只包含一个扫描仪的数据)/"
ther=0.5
label_path=path+"/csv"
pre_point_path="infer_save/Ablation_k=6/pre_point/"

label_names=  os.listdir(label_path)
num=0
tp=0
fp=0
a=0
for label_name in label_names:
  
    label_file=label_path+'/'+label_name
    with open(label_file,encoding='utf-8')as f:
        reader = csv.reader(f)
        for point in reader:
            num=num+1
pre_names=  os.listdir(pre_point_path)
for pre_name in tqdm(pre_names):
    pre_file=pre_point_path+'/'+pre_name
    gt_file=label_path+'/'+pre_name[:-4]+'.csv'
    pre_file = pd.read_table(pre_file,header = None,names=list('a'))
    for j in range(len(pre_file)):         #遍历每一行
        coordinate_input=pre_file['a'][j].split()
        y=int(coordinate_input[1])
        x=int(coordinate_input[0])
        prob=float(coordinate_input[2])
        temp=1
        if(prob>=ther):
            a=a+1
            try:
                with open(gt_file,encoding='utf-8')as f:
                    reader = csv.reader(f)
                    ##################################################
                    # if len(reader)==0:
                    #     continue
                    for point in reader:
                        x_gt=int(point[0])
                        y_gt=int(point[1])
                        if((x-x_gt)**2+(y-y_gt)**2<900):
                            tp=tp+1
                            temp=0
                            break
            except:
                test=1
            if(temp):
                fp=fp+1
p=tp/(tp+fp)
r=tp/num
f1=p*r*2/(p+r)
print("tp:"+str(tp))
print("fp:"+str(fp))
print("p:"+str(p))
print("r:"+str(r))
print("f1:"+str(f1))