import numpy as np
import os
from lib.Network_MCR_PVT import Network
from test.test_data import test_dataset
from test.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm
from test.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
dataset_path = '/data/clb/Revised_MCRNet/FSEL/FSEL_ECCV_2024/data/test/' 

# root_model_path = '/data/clb/Revised_MCRNet/Change_backbone/OtherMethod/'
# root_model_path = '/data/clb/Revised_MCRNet/Change_backbone/Ablation/'
# root_model_path = '/data/clb/Revised_MCRNet/Change_backbone/Our_ResNet/'
# root_model_path = '/data/clb/Revised_MCRNet/Change_backbone/cccc/'
# root_model_path = '/data/clb/Revised_MCRNet/Change_backbone/save_try_preds/'


root_model_path = '/data/clb/mamba_capsule/save_try_preds/' 

# model_names = [d for d in os.listdir(root_model_path) if os.path.isdir(os.path.join(root_model_path, d))]
# model_name = 'MCRNet-S'  

# if 'EVP' in model_names:
#     model_names.remove('EVP')
#     model_names.remove('RMGL')
#     model_names.remove('UGTR')

# test_datasets = ['CAMO','COD10K','NC4K']  #
# test_datasets = ['COD10K','NC4K']  #          
# test_datasets = ['NC4K']  #
test_datasets = ['CAMO']  #

with open('metrics.txt', 'a') as f:
    for model_name in ['PVT']:
        
        for dataset in test_datasets:
            # sal_root = dataset_path_pre +dataset+'/'
            sal_root = root_model_path + str(model_name)+ '/'+dataset

            gt_root = dataset_path +dataset+'/GT/'
            test_loader = test_dataset(sal_root, gt_root)
            wfm= cal_wfm()

            mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
            for i in range(test_loader.size):
                print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = gt.size
                    sal = sal.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res)
                if res.max() == res.min():
                    res = res/255
                else:
                    res = (res - res.min()) / (res.max() - res.min())
                mae.update(res, gt)
                sm.update(res,gt)
                fm.update(res, gt)
                em.update(res,gt)
                wfm.update(res,gt)

            MAE = mae.show()
            maxf,meanf,_,_ = fm.show()
            sm = sm.show()
            em = em.show()
            wfm = wfm.show()
            print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em))
            f.write(f'{model_name}:  dataset: {dataset}  MAE: {MAE:.4f}  maxF: {maxf:.4f}  avgF: {meanf:.4f}  wfm: {wfm:.4f}  Sm: {sm:.4f}  Em: {em:.4f}\n')







        