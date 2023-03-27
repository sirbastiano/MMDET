# Config my baseline with 
import os
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from utils import *

warnings.filterwarnings("ignore")
plt.style.use(['science','nature','no-latex'])
torch.cuda.empty_cache()
os.system('clear')
cwd = os.getcwd()

models = ['retina18','retina50','retina101','mask50','mask101','cascade_mask50','cascade_mask101',
          'detr','centripetalnet','vfnet','efficientdet','sparce_rcnn50','htc50','htc101','fcos50',
          'hrnet40_cascade','ssd_vgg16','fovea50','fovea101','centernet18','tridentnet50','fsaf50',
          'retina50_timm','retina_swin']

classes = ('Wake',)

Dataselector = ['SARWake'] # For multiple datasets

if __name__ == '__main__':
     # Select Model and specify if to train and test
     train_bool = True
     test_bool = True
     # Hyper-Parameters:
     # max_epochs = 25
     for selection in ['retina50','retina101','hrnet40_cascade','ssd_vgg16'   ]:
          for dataset in Dataselector:
               for lr, max_epochs in zip([0.002,0.0015,0.0005],[25,50,75]):
                    for lr_schedule in ['step','CosineAnnealing']: # "CosineAnnealing" or "step"
                         load_from = None
                         img_size=768 # Specify in the dataset, here won't be collected!
                         data_root = "data/DATASETS/SARWake"

                         workdir = f'checkpoints/{dataset}/{selection}/{selection}_'+getCurrentTime()+f'_{img_size}_{max_epochs}e_{dataset}_lr_{lr}_{lr_schedule}'
                         extra_args = {'max_epochs':max_epochs, 'lr':lr, 'load_from':load_from, 'lr_cfg':lr_schedule,
                                        'data_root':data_root, 'classes':classes} # extra_args = None # None to use default values.


                         if train_bool:
                              train(selection=selection, workdir=workdir, extra_args=extra_args)
                         if test_bool:
                              # To constrain into a specific path:
                              # workdir = 'checkpoints/B3/cascade_mask50/cascade_mask50_2022-07-28T11:01:26_768_50e_B3_lr_0.001_step'
                              test(workdir=workdir)
                         
                         plotRes(workdir, title=selection+f'_{dataset}_')
                         keepGoodWeight(workdir)

