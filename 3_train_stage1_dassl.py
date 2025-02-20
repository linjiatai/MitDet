import argparse
import logging
import math
import os
import random
import shutil
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tool import pyutils

from dataset.cifar import *
import models.resnet38d as models

logger = logging.getLogger(__name__)
best_acc = 0

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

def main():
    print('-------------------------phase 1 training-------------------------')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--max_epoches", default=200, type=int)
    parser.add_argument('--batch_size', default=400, type=int,
                        help='train batchsize')
    parser.add_argument('--num-workers', type=int, default=10, help='number of workers')
    parser.add_argument('--arch', default='resnet38', type=str, choices=['resnet38'], help='dataset name')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument("--network", default="network.resnet38_cls_dassl", type=str)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs ')
    parser.add_argument('--cluster_k', default=4, type=int)
    args = parser.parse_args()

    args.p_labeled_root = 'DATA/MIDOG2021/dataset_step1/train/p/'
    args.n_labeled_root = 'DATA/MIDOG2021/dataset_step1/train/n_balance/'
    args.test_root = 'DATA/MIDOG2021/dataset_step1/val/'


    args.num_classes =2
    save_path = 'save_weights/stage1/k1='+str(args.cluster_k)+'/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    model = getattr(importlib.import_module(args.network), 'Net')(n_class=args.num_classes)
    weights_dict = models.convert_mxnet_to_torch('init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
    model.load_state_dict(weights_dict, strict=False)
    logger.info("成功加载resnet38")
    model = model.cuda()
    #优化器
    optimizer=optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wt_dec)
    #测试数据
    test_dataset=get_mito_test(args,args.test_root)
    test_loader = DataLoader(
                    test_dataset,
                    sampler=SequentialSampler(test_dataset),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)   
    best_f1 = 0
    for ep in range(args.max_epoches):
        model.train()
        #训练数据导入，阳性样本和阴性样本按比例混合导入
        train_dataset=get_mito_train(args.p_labeled_root,args.n_labeled_root)
        train_sampler = RandomSampler
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler(train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)
        # train_iter = iter(train_loader)
        avg_meter = pyutils.AverageMeter(
            'loss',
            )
        for iter, (input_names, inputs_x, targets_x) in enumerate(train_loader):
            batch_size = inputs_x.shape[0]
            inputs = inputs_x.cuda()
            targets_x = targets_x.cuda()
            logits,feature,y = model(inputs)
            # logits_x = logits[:batch_size]
            # del logits
            _, targets_x_one = torch.max(targets_x, dim=-1)
            Lx = F.cross_entropy(logits, targets_x_one)
            loss = Lx
            avg_meter.add({'loss':loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        #打印训练信息                   
        print('Epoch:%2d' % (ep),
                'Loss:%.4f' % (avg_meter.get('loss')),
                'lr: %.5f' % (optimizer.param_groups[0]['lr'])     
        )    
        if (ep>=0):   
            ###val
            model.eval()
            tp=0
            tn=0
            fp=0
            fn=0
            with torch.no_grad():
                for batch_idx, (_,inputs, targets) in enumerate(test_loader):

                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    if args.arch=='resnet38':
                        x,feature,y= model(inputs.cuda())
                    elif args.arch=='r50':
                        x= model(inputs.cuda())
                    else:
                        x= model(inputs)
                    
                    prob = torch.softmax(x,dim=-1).cpu().data.numpy()
                    gt = targets.cpu().data.numpy()
                    for num, one in enumerate(prob):
                    
                        pass_cls = 0 if one[0]>0.5 else 1
                        true_cls = np.where(gt[num] == 1)[0]

                        if(pass_cls==0 and true_cls==0):
                            tp=tp+1
                        if(pass_cls==0 and true_cls==1):
                            fp=fp+1
                        if(pass_cls==1 and true_cls==0):
                            fn=fn+1
                        if(pass_cls==1 and true_cls==1):
                            tn=tn+1
            try:
                p=tp/(tp+fp)
                r=tp/(tp+fn)
                f1=2*p*r/(p+r)
                print("tp:"+str(tp))
                print("fp:"+str(fp))
                print("fn:"+str(fn))
                print("p:"+str(p))
                print("r:"+str(r))
                print("f1:"+str(f1))
                if f1> best_f1:
                    best_f1=f1
                    torch.save(model.state_dict(), save_path+'/ep'+str(ep)+'_f1_'+str(format(f1,".3f"))+'_p_'+str(format(p,".3f"))+'_r_'+str(format(r,".3f"))+"_tp_"+str(tp)+"_fp_"+str(fp)+"_fn_"+str(fn)+'.pth') 
                    torch.save(model.state_dict(), save_path+'/best_model.pth') 
            except:
                print('p=0')

if __name__ == '__main__':
    main()