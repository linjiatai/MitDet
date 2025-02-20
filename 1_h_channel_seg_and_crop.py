import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import csv
from PIL import Image,ImageEnhance
import os.path
import cv2
from torchvision import transforms
from tqdm import tqdm
from skimage import morphology, measure
import spams
import shutil
import random
STAIN_NUM=2
THRESH=0.9
LAMBDA1=0.01
LAMBDA2=0.01
ITER=100
fast_mode=0
getH_mode=0
transform = transforms.Compose([transforms.ToTensor()]) 
class InferDataset():
    def __init__(self, image_file,data_points,transform,width,highth):
        self.image=Image.open(image_file)
        self.data_points = data_points
        self.transform=transform
        self.width=width
        self.highth=highth

    def __getitem__(self, index):
        [x_center,y_center]=self.data_points[index]
        box = (x_center-self.width//2, y_center-self.highth//2, x_center+self.width//2, y_center+self.highth//2)
        img = self.image.crop(box).convert('RGB')
        img=transform(img)
        return  x_center,y_center,img
        
    def __len__(self):
        return len(self.data_points)
def getV(img):
    
    I0 = img.reshape((-1,3)).T
    I0[I0==0] = 1
    V0 = np.log(255 / I0)

    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = img_LAB[:, :, 0] / 255 < THRESH
    I = img[mask].reshape((-1, 3)).T
    I[I == 0] = 1
    V = np.log(255 / I)
    return V0, V

def getW(V):
    W = spams.trainDL(np.asfortranarray(V), K=STAIN_NUM, lambda1=LAMBDA1, iter=ITER, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
    W = W / np.linalg.norm(W, axis=0)[None, :]
    if (W[0,0] < W[0,1]):
        W = W[:, [1,0]]
    return W


def getH(V, W):
    if (getH_mode == 0):
        H = spams.lasso(np.asfortranarray(V), np.asfortranarray(W), mode=2, lambda1=LAMBDA2, pos=True, verbose=False).toarray()
    elif (getH_mode == 1):
        H = np.linalg.pinv(W).dot(V)
        H[H<0] = 0
    else:
        H = 0
    return H

def stain_separate(img):
    if (fast_mode == 0):
        V0, V = getV(img)
        W = getW(V)
        H = getH(V0, W)
    elif (fast_mode == 1):
        m = img.shape[0]
        n = img.shape[1]
        grid_size_m = int(m / 5)
        lenm = int(m / 20)
        grid_size_n = int(n / 5)
        lenn = int(n / 20)
        W = np.zeros((81, 3, STAIN_NUM)).astype(np.float64)
        for i in range(0, 4):
            for j in range(0, 4):
                px = (i + 1) * grid_size_m
                py = (j + 1) * grid_size_n
                patch = img[px - lenm : px + lenm, py - lenn: py + lenn, :]
                V0, V = getV(patch)
                W[i*9+j] = getW(V)
        W = np.mean(W, axis=0)
        V0, V = getV(img)
        H = getH(V0, W)
    return W, H

def main():


    # subset_tags = ['train', 'val']
    subset_tags = ['val']

    for subset_tag in subset_tags:
    
        image_path="DATA/MIDOG2021/0_data_ori/"+subset_tag+"/png/"
        label_gt_path="DATA/MIDOG2021/0_data_ori/"+subset_tag+"/csv/"

        output_path_p="DATA/MIDOG2021/dataset_step1/"+subset_tag+"/p"
        output_path_n="DATA/MIDOG2021/dataset_step1/"+subset_tag+"/n"
        if os.path.exists(output_path_p):
            shutil.rmtree(output_path_p)
        if os.path.exists(output_path_n):
            shutil.rmtree(output_path_n)    
        os.makedirs(output_path_p, exist_ok=True)
        os.makedirs(output_path_n, exist_ok=True)

        width=80
        highth=80
        width2=60
        highth2=60
        width3=160
        highth3=160

        patch_names_list= os.listdir(image_path)
        num_positive = 0
        num_sample_p = 0
        for patch_name in tqdm(patch_names_list):
            infer_points_list=[]
            basename=patch_name[:-4]
            id = basename.split('_')[1]
            label_file=label_gt_path+'/'+basename+'.csv'
            img = Image.open(image_path+'/stain1_'+id+'.png')
            [h,w] = img.size
            h_channel = np.array(Image.open('DATA/MIDOG2021/0_h_channels/'+id+'.png'))
            label_image =measure.label(h_channel)
            regions=measure.regionprops(label_image)
            points=[]
            with open(label_file,encoding='utf-8')as fp:
                reader = csv.reader(fp)
                # 遍历数据
                for point in reader:
                    points.append(point)
            num_positive += len(points)
            rand=random.random()
            for region in regions: #循环得到每一个连通区域属性集
            #忽略小区域
                area=region.area
                if area < 5 or area>2500:
                    continue
                y=(region.bbox[0]+region.bbox[2])//2
                x=(region.bbox[1]+region.bbox[1])//2

                if (x-width//2<0) or (y-highth//2<0) or (x+width//2>w) or (y+highth//2>h):
                    continue

                point=[x,y]
                if point in infer_points_list:
                    continue
                infer_points_list.append(point)
                is_p = 0
                box = [x-width//2, y-highth//2,x+width//2, y+highth//2]
                for point in points:
                    if (int(point[0])>x-width2//2 and int(point[0])<x+width2//2 \
                        and int(point[1])>y-highth2//2 and int(point[1])<y+highth2//2):
                        is_p=1
                        num_sample_p += 1
                        new_image = img.crop(box)
                        new_image.save(output_path_p+'/stain1_'+id+'_'+str(int(x))+'_'+str(int(y))+'[10]'+'.png')
                if (int(point[0])>x-width3//2 and int(point[0])<x+width3//2 \
                        and int(point[1])>y-highth3//2 and int(point[1])<y+highth3//2):
                    is_p=1
                    continue
                rand = random.random()
                if (is_p==0) and (rand<=0.1):
                    new_image = img.crop(box)
                    new_image.save(output_path_n+'/stain1_'+id+'_'+str(int(x))+'_'+str(int(y))+'[01]'+'.png') 
            print('Number of positive annotation : ', num_positive)
            print('Number of positive samples : ', num_sample_p)

if __name__ == '__main__':
    main()