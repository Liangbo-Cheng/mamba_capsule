import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2

from lib.Network_MCR_Vmamba_T import Network  

from utilss.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', default=352, type=int, help='network input size')
parser.add_argument('--pth_path', type=str, default='/data1/MCRNet/best.pth') 
parser.add_argument('--test_dataset_path', type=str, default='/data1/data/test')
parser.add_argument('--encoder_dim', default=[96,192,384,768], type=int, help='dim of each encoder layer')
parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
parser.add_argument('--dim', default=64, type=int, help='dim')

opt = parser.parse_args()

for _data_name in ['CAMO','COD10K','NC4K']:

    data_path = opt.test_dataset_path+'/{}/'.format(_data_name)

    save_path = './preds_failswint/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)     
    
    os.makedirs(save_path, exist_ok=True)

    model = Network(args=opt)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.img_size)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        print('> {} - {}'.format(_data_name, name))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        result = model(image)
        result = result[2][3]
        res = F.interpolate(result, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name,res*255)
