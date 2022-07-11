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
          'detr','centripetalnet','vfnet','efficientdet','sparce_rcnn',
          'hrnet40_cascade',]

# bands = ['B2','B3','B4','B8']
bands = ['B2']

if __name__ == '__main__':
     # Select Model and specify if to train and test
     train_bool = True
     test_bool = True
     # Hyper-Parameters:
     max_epochs = 100
     # band = 'B2'
     for selection in ['retina18']:
          for band in bands:
               lr = 0.001 * 0.5 
               lr_schedule = 'CosineAnnealing' # "CosineAnnealing" or "linear"
               load_from = "checkpoints/B2/retina18/retina18_2022-07-11T19:19:53_768_20e_B2_lr_0.0005_Cosine/epoch_20.pth"
               img_size=768 # Specify in the dataset, here won't be collected
               data_root = band_selector(band)

               workdir = f'checkpoints/{band}/{selection}/{selection}_'+getCurrentTime()+f'_{img_size}_{max_epochs}e_{band}_lr_{lr}_Cosine'
               extra_args = {'max_epochs':max_epochs, 'lr':lr, 'load_from':load_from, 'lr_cfg':lr_schedule,
                              'data_root':data_root}
               # extra_args = None # Pass None to use default values.


               if train_bool:
                    train(selection=selection, workdir=workdir, extra_args=extra_args)
               if test_bool:
                    test(selection=selection, workdir=workdir)
               
               plotRes(workdir, title=selection+f'_{band}_')
               keepGoodWeight(workdir)



