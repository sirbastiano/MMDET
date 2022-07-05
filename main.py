# Config my baseline with 
import os
import torch
from utils import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(['science','nature','no-latex'])

torch.cuda.empty_cache()

os.system('clear')
cwd = os.getcwd()

models = ['retina','mask50','mask101','cascade','detr','centripetalnet','vfnet','efficientdet','sparce_rcnn',
          'hrnet40',]


if __name__ == '__main__':

     traintest = True
     selection = 'hrnet40'
     workdir = 'checkpoints/'+selection + getCurrentTime()+'_512_B4'
     if traintest:
          train(selection=selection, workdir=workdir)
          test(selection=selection, workdir=workdir)

     plotRes(workdir, title=selection+'_B4_')

