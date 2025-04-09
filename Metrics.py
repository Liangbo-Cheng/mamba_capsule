import numpy as np
import os
from lib.Network_MCR_Vmamba_T import Network
from test.test_data import test_dataset
from utils.dataloader import EvalDataset
from utils.evaluator import Eval_thread
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dataset_path = '/data1/data/test/' 
root_model_path = '/data1/MCRNet/preds/' 
test_datasets = ['CAMO','COD10K','NC4K']  #
for model_name in ['']:  
    threads = []
    for dataset in test_datasets:
        sal_root = root_model_path+ str(model_name)+ '/'+dataset+ '/'        
        gt_root = dataset_path +dataset+'/GT/'
        test_loader = test_dataset(sal_root, gt_root)
        loader = EvalDataset(sal_root, gt_root, 'CODRGB')
        thread = Eval_thread(loader, model_name, dataset, './', cuda=True)
        threads.append(thread)
    for thread in threads:
        print(thread.run())

