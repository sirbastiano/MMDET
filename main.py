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

models = ['retina','mask50','mask101','cascade_mask50','detr','centripetalnet','vfnet','efficientdet','sparce_rcnn',
          'hrnet40_cascade',]

if __name__ == '__main__':
     # Select Model and specify if to train and test
     traintest = False
     optimize_anchor = True
     selection = 'cascade_mask50'
     # Hyper-Parameters:
     max_epochs = 20
     lr = 0.01
     load_from = 'checkpoints/hrnet402022-07-06T01:06:10_640_40e_B4_lr_0.01/epoch_40.pth'
     img_size=712
     band = 'B4'
     data_root = band_selector(band)


     workdir = 'checkpoints/'+selection + getCurrentTime()+f'_{img_size}_{max_epochs}e_{band}_lr_{lr}'
     extra_args = {'max_epochs':max_epochs, 'lr':lr, 'load_from':load_from, 
                    'img_size':img_size, 'data_root':data_root}
     # extra_args = None # Pass None to use default values.

     if optimize_anchor:
          os.system(f'python tools/analysis_tools/optimize_anchors.py {selector(selection)} \
                         --algorithm differential_evolution\
                         --input-shape 712 712 \
                         --output-dir {workdir}')

     if traintest:
          train(selection=selection, workdir=workdir, extra_args=extra_args)
          test(selection=selection, workdir=workdir)
          plotRes(workdir, title=selection+f'_{band}_')

