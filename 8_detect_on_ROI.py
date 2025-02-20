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
from PIL import Image
import os.path
import cv2
from torchvision import transforms
from h_channel import *
import math
from tqdm import tqdm
from skimage import measure,morphology

class InferDataset():
    def __init__(self, image_file,data_points,transform,width,highth):
        img=cv2.imread(image_file)
        self.image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.data_points = data_points
        self.transform=transform

    def __getitem__(self, index):
        [x_center,y_center]=self.data_points[index]
        img=self.image[x_center-width//2:x_center+width//2,y_center-highth//2:y_center+highth//2,:]
        img=transform(img)
        return  x_center,y_center,img
        
    def __len__(self):
        return len(self.data_points)

n_class=2
cluster_k = 4
cluster_k_2 = 6
num_workers=16
batch_size=6
thor=0.5
# weights2="/home/linjiatai/有丝分裂_linjiatai/2_训练/data_linjiatai/step_2/k6_cluster/ck/ep80_f1_0.767_r_0.829_p_0.713_tp_1045_fp_420_fn_215.pth"
weights2="/home/linjiatai/有丝分裂_linjiatai/2_训练/data_linjiatai/step_2/k2_cluster/ck/ep53_f1_0.762_r_0.844_p_0.694_tp_1064_fp_470_fn_196.pth"

path="/home/linjiatai/MitDet_linjiatai（最终）/DATA/MIDOG2022(只包含一个扫描仪的数据)/"
image_path=path+"/png"
label_gt_path=path+"/csv"
savepath="infer_save/Ablation_k=2_22/pre/"
txt_output="infer_save/Ablation_k=2_22/pre_point/"
pre_h_mask='infer_save/Ablation_k=2_22/pre_h_mask/'

os.makedirs(savepath, exist_ok=True)
os.makedirs(txt_output, exist_ok=True)
os.makedirs(pre_h_mask, exist_ok=True)

width=80
highth=80
width2=80
highth2=80

model2 = getattr(importlib.import_module("network.resnet38_cls_dassl_multi_label"), 'Net')(1, num_subcategory=cluster_k_2*2)
weights = torch.load(weights2)
# del weights['fc8_200.weight']
model2.load_state_dict(weights, strict=False)
model2.eval()
model2.cuda()

tp=0
num=0

patch_names_list= os.listdir(image_path)

transform = transforms.Compose([transforms.ToTensor()]) 
for patch_name in tqdm(patch_names_list):
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
    id = image_file.split('/')[-1].split('.')[0].split('_')[-1]
    # h_channel = np.array(Image.open('DATA/MIDOG2021/0_h_channels/'+id+'.png'))

    #消除小的连通域
    # import morphology
    h_channel=morphology.remove_small_objects(h_channel,min_size=10,connectivity=1)

    label_image =measure.label(h_channel)#label函数得到连通域

    temp=Image.fromarray((h_channel*255).astype('uint8'))
    temp.save(pre_h_mask+"/"+patch_name)

    regions=measure.regionprops(label_image)
    # img=Image.open(image_file)
    for region in regions: #循环得到每一个连通区域属性集
        #由于下采样了，要*2
        x_center=(region.bbox[0]+region.bbox[2])//2
        y_center=(region.bbox[1]+region.bbox[3])//2

        if x_center-width//2<0 :
            x_center=width//2
        if y_center-highth//2<0 :
            y_center=highth//2
        if x_center+width//2>w :
            x_center=w-width//2
        if y_center+highth//2>h:
            y_center=h-highth//2

        # detect_img=img[x_center-56:x_center+56,y_center-56:y_center+56,:]
        point=[x_center,y_center]
        if point in infer_points_list:
            continue
        infer_points_list.append(point)

        # detect_img=detect_img.unsqueeze(0)#将图片维度变为[1,3,112,112]
    infer_dataset = InferDataset(image_file,infer_points_list,transform=transform,width=width,highth=highth)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=False,
                                batch_size=batch_size
                                )
    torch.cuda.empty_cache()
    
    for iter,  (x_center,y_center,img_list) in enumerate(infer_data_loader):   
        output2, _, _,_,_ = model2(img_list.cuda(),1) 
        prob2 = torch.softmax(output2,dim=-1).cpu().data.numpy()
        for (x,y,img_,pre) in zip(x_center,y_center,img_list,prob2): 
            if pre[0]>thor:  
                img_=img_.unsqueeze(0)
                g_cam=model2.forward_cam(img_.cuda())
                g_cam=g_cam.squeeze(0)[0]
                if x-width2//2<0 :
                    x=width2//2
                if y-highth2//2<0 :
                    y=highth2//2
                if x+width2//2>w:
                    x=w-width2//2
                if y+highth2//2>h:
                    y=h-highth2//2

                norm_cam = np.array(g_cam.cpu().detach())
                _range = np.max(norm_cam) - np.min(norm_cam)
                norm_cam = (norm_cam - np.min(norm_cam))/_range
                norm_cam=norm_cam*255
                norm_cam=cv2.resize(norm_cam,(width,highth))
                output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2]=np.maximum(norm_cam,\
                            output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2])
                prob_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2]=np.maximum(pre[0],\
                            prob_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2])
    output_map=output_map>200
    output_map=morphology.remove_small_objects(output_map,min_size=25,connectivity=1)

    output_map=output_map*255
    # cv2.imwrite("1.png", output_map)   
    
    label_output =measure.label(output_map)
    regions_output=measure.regionprops(label_output)
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for region_output in regions_output:
        num=num+1
        y_min=region_output.bbox[0]
        y_max=region_output.bbox[2]
        x_min=region_output.bbox[1]
        x_max=region_output.bbox[3]
        x=(x_min+x_max)//2
        y=(y_min+y_max)//2
        prob=prob_map[(y_min+y_max)//2,(x_min+x_max)//2]
        p1=(x-40,y-40)
        p2=(x+40,y+40)
        color_box=(0,0,255)

        #保存预测结果

        line=(int(x),int(y),prob)
        with open(txt_output +'/'+basename+ '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
        #可视化
        try:
            with open(label_file,encoding='utf-8')as fp:
                reader = csv.reader(fp)
                for point in reader:
                    if((int(point[0])-x)**2+(int(point[1])-y)**2<900):
                        color_box=(0,255,0)
                        tp=tp+1
                        break
        except:
            test=1
        cv2.rectangle(img, p1, p2, color_box, 5, cv2.LINE_AA)
        cv2.putText(img,str(format(prob,".2f")),p1,0,2,color=color_box, thickness=3)#图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(savepath+'/'+basename+'.png', img.astype('uint8'))
print("tp:"+str(tp))
print("num:"+str(num))
    








    