from email.mime import base
import os
from cv2 import norm
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam,GradCam2
import matplotlib.pyplot as plt
from h_channel import *
import shutil
transform = transforms.Compose([transforms.ToTensor()]) 
width=80
highth=80
width2=80
highth2=80
class InferDataset():
    def __init__(self, image_file,data_points,transform,width,highth):
        img=cv2.imread(image_file)
        self.image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.img = Image.open(image_file)
        self.width=width
        self.highth=highth
        self.data_points = data_points
        self.transform=transform

    def __getitem__(self, index):
        [x_center,y_center]=self.data_points[index]
        # box = (x_center-self.width//2, y_center-self.highth//2, x_center+self.width//2, y_center+self.highth//2)
        # img = self.img.crop(box)
        img=self.image[x_center-width//2:x_center+width//2,y_center-highth//2:y_center+highth//2,:]
        img=transform(img)
        return  x_center,y_center,img
        
    def __len__(self):
        return len(self.data_points)

def main():
    """生成第二阶段的训练数据和al数据"""
    n_class=2
    weights='save_weights/stage1/k1=4/ep162_f1_0.893_p_0.841_r_0.954_tp_390_fp_74_fn_19.pth'

    # datatags = ['train', 'val']
    datatags = ['train']

    for datatag in datatags:

        image_path='DATA/MIDOG2021/0_data_ori/'+datatag+'/png/'
        label_gt_path='DATA/MIDOG2021/0_data_ori/'+datatag+'/csv/'
        output_path_p = 'DATA/MIDOG2021/dataset_best/'+datatag+'/p/'
        output_path_n = 'DATA/MIDOG2021/dataset_best/'+datatag+'/n/'

        if os.path.exists(output_path_n):
            shutil.rmtree(output_path_n)
        os.makedirs(output_path_n)
        if os.path.exists(output_path_p):
            shutil.rmtree(output_path_p)
        os.makedirs(output_path_p)
    
        model = getattr(importlib.import_module("network.resnet38_cls_dassl"), 'Net')(n_class=n_class)
        model.load_state_dict(torch.load(weights), strict=False)
        model.eval()
        model.cuda()

        tp=0
        num=0

        ffmm = model.bn7

        patch_names_list= os.listdir(image_path)
        grad_cam = GradCam2(model=model, feature_module=ffmm, \
                    target_layer_names=["1"], use_cuda=True)
        
        for patch_name in tqdm(patch_names_list):
            num=0
            infer_points_list=[]
            basename=patch_name[:-4]
            image_file=image_path+'/'+patch_name
            label_file=label_gt_path+'/'+basename+'.csv'
            try:
                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print(image_file)
                continue
            [w,h] = img.shape[:2]
            output_map=np.zeros((w,h))
            prob_map=np.zeros((w,h))
            img_temp=cv2.pyrDown(img)
            img_temp = cv2.GaussianBlur(img_temp,(3,3),0)
            
            w_channel,h_channel=stain_separate(img_temp)
            h_channel=h_channel[0,:]
            h_channel=h_channel.reshape(math.ceil(w/2),math.ceil(h/2))
            # h_channel=h_channel>2
            h_channel=h_channel>np.max(h_channel)*0.4

            h_channel=morphology.remove_small_objects(h_channel,min_size=15,connectivity=1)
            label_image =measure.label(h_channel)#label函数得到连通域 
            regions=measure.regionprops(label_image)
            for region in regions: #循环得到每一个连通区域属性集
                
                x_center=(region.bbox[0]+region.bbox[2])
                y_center=(region.bbox[1]+region.bbox[3])
                

                if (x_center-width//2<0) or (y_center-highth//2<0) or  (x_center+width//2>w) or (y_center+highth//2>h):
                    continue
                point=[x_center,y_center]
                infer_points_list.append(point)

            infer_dataset = InferDataset(image_file,infer_points_list,transform=transform,width=width,highth=highth)
            infer_data_loader = DataLoader(infer_dataset,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=False,
                                        batch_size=200
                                        )
            torch.cuda.empty_cache()

            for iter,  (x_center,y_center,img_list) in enumerate(infer_data_loader):   
                grayscale_cam, output = grad_cam(img_list, target_category=0)
                output,_,_=model(img_list.cuda())
                prob = torch.softmax(output,dim=-1).cpu().data.numpy()
                for (x,y,g_cam, pre) in zip(x_center,y_center,grayscale_cam,prob):
                    
                    if(pre[0]>=0.5):
                        num=num+1
                        norm_cam = np.array(g_cam)
                        _range = np.max(norm_cam) - np.min(norm_cam)
                        norm_cam = (norm_cam - np.min(norm_cam))/_range
                        norm_cam=norm_cam*255
                        output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2]=np.maximum(norm_cam,\
                                    output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2])


            output_map=output_map>200
            output_map=morphology.remove_small_objects(output_map,min_size=100,connectivity=1)
            output_map=output_map*255
            label_output =measure.label(output_map)
            regions_output=measure.regionprops(label_output)

            points=[]
            with open(label_file,encoding='utf-8')as fp:
                reader = csv.reader(fp)
                # 遍历数据
                for point in reader:
                    points.append(point)

            img=Image.open(image_file)
            img_stain2 = Image.open('DATA/MIDOG2021/0_stains_ROIs/png/'+image_file.split('/')[-1].replace('stain1', 'stain2'))
            img_stain3 = Image.open('DATA/MIDOG2021/0_stains_ROIs/png/'+image_file.split('/')[-1].replace('stain1', 'stain3'))
            basename2 = image_file.split('/')[-1].replace('stain1', 'stain2').split('.')[0]
            basename3 = image_file.split('/')[-1].replace('stain1', 'stain3').split('.')[0]
                #得到二阶段的训练数据
            for region_output in regions_output:
                y_min=region_output.bbox[0]
                y_max=region_output.bbox[2]
                x_min=region_output.bbox[1]
                x_max=region_output.bbox[3]
                p1=(x_min,y_min)
                p2=(x_max,y_max)
                x=(x_min+x_max)//2
                y=(y_min+y_max)//2

                if (x-width2//2<0) or (y-highth2//2<0) or (x+width2//2>w) or (y+highth2//2>h):
                    continue
                box = (x-width2//2, y-highth2//2, x+width2//2, y+highth2//2)
                region = img.crop(box)
                new_image = Image.new('RGB', (width2,highth2), (0, 0, 0))
                new_image.paste(region,(0,0))
                new_image_stain2 = img_stain2.crop(box)
                new_image_stain3 = img_stain3.crop(box)
                temp=1
                try:
                    with open(label_file,encoding='utf-8')as fp:
                        reader = csv.reader(fp)
                        # 遍历数据
                        for point in reader:
                            if(int(point[0])>x-width2//2 and int(point[0])<x+width2//2 \
                            and int(point[1])>y-highth2//2 and int(point[1])<y+highth2//2):
                                temp=0
                                if(int(point[0])>x-width2//2+10 and int(point[0])<x+width2//2-10 \
                                and int(point[1])>y-highth2//2+10 and int(point[1])<y+highth2//2-10):
                                    new_image.save(output_path_p+'/'+basename+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[10]'+'.png') 
                                    new_image_stain2.save(output_path_p+'/'+basename2+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[10]'+'.png') 
                                    new_image_stain2.save(output_path_p+'/'+basename3+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[10]'+'.png') 
                                    tp=tp+1
                                    break
                    if(temp):
                        new_image.save(output_path_n+'/'+basename+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png')
                        new_image_stain2.save(output_path_n+'/'+basename2+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png') 
                        new_image_stain2.save(output_path_n+'/'+basename3+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png')  
                except:
                    new_image.save(output_path_n+'/'+basename+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png')
                    new_image_stain2.save(output_path_n+'/'+basename2+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png') 
                    new_image_stain2.save(output_path_n+'/'+basename3+'_'+str(int(x-width2//2))+'_'+str(int(y-highth2//2))+'[01]'+'.png')  
        print("tp:"+str(tp))
                    
if __name__ == '__main__':
    main()