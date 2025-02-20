import argparse
import os
import shutil
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cluster_k', default=2, type=int)
    args = parser.parse_args()

    args.save_p = 'DATA/MIDOG2021/dataset/train/multi_stain_k='+str(args.cluster_k)+'/p/'
    args.save_n = 'DATA/MIDOG2021/dataset/train/multi_stain_k='+str(args.cluster_k)+'/n/'

    if os.path.exists('DATA/MIDOG2021/dataset/train/multi_stain_k='+str(args.cluster_k)+'/'):
        shutil.rmtree('DATA/MIDOG2021/dataset/train/multi_stain_k='+str(args.cluster_k)+'/')
    os.makedirs(args.save_p)
    os.makedirs(args.save_n)

    args.data_p = 'DATA/MIDOG2021/dataset/train/stain1/p/'
    args.data_n = 'DATA/MIDOG2021/dataset/train/stain1/n_step3-4_sampling_k='+str(args.cluster_k)+'/'

    files_p = os.listdir(args.data_p)
    files_n = os.listdir(args.data_n)

    for file in tqdm(files_n):
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain1/n_step1_hchannel/'+file, dst=args.save_n+'/'+file)
        file_stain2 = file.replace('stain1', 'stain2')
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain2/n/'+file_stain2, dst=args.save_n+'/'+file_stain2)
        file_stain3 = file.replace('stain1', 'stain3')
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain3/n/'+file_stain3, dst=args.save_n+'/'+file_stain3)
    for file in tqdm(files_p):
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain1/p/'+file, dst=args.save_p+'/'+file)
        file_stain2 = file.replace('stain1', 'stain2')
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain2/p/'+file_stain2, dst=args.save_p+'/'+file_stain2)
        file_stain3 = file.replace('stain1', 'stain3')
        shutil.copy(src='DATA/MIDOG2021/dataset/train/stain3/p/'+file_stain3, dst=args.save_p+'/'+file_stain3)
    
   
if __name__ == '__main__':
    main()