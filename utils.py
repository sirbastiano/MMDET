import os
import torch
from utils import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use(['science','nature','no-latex'])

torch.cuda.empty_cache()

os.system('clear')
cwd = os.getcwd()

def test(selection:str, workdir:str):
     """Executes the test on the selected model.
          Input: selection -> the model selected
                 wordir -> the directory where the weights are stored
          Outputs: saving.pkl and json evals of the model
     """
     config_path = selector(selection)

     weights = getListWeight(workdir)
     for idx, checkpoint in enumerate(weights):
          outfile = workdir +f'/saving_{idx}.pkl'
          eval_type = 'bbox'
          checkpoint = checkpoint.absolute().as_posix()
          print(checkpoint)
          os.system(f'python tools/test.py {config_path} {checkpoint} --work-dir {workdir} --out {outfile} --eval {eval_type}')


def train(selection: str, workdir):
     """Executes the test on the selected model.
          Input: selection -> the model selected
                 wordir -> the directory where the weights are saved
          Outputs: epoch.h5 
     """
     config_path = selector(selection)
     
     MAX_TRIES = 1 # max number of training retrials
     count = 0
     while count < MAX_TRIES:
          try:
               # os.system(f'python tools/train.py {config_path} --auto-resume --seed 0')
               os.system(f'bash tools/dist_train.sh {config_path} 1 --auto-resume --seed 0 --cfg-options work_dir={workdir}')
               print("Execution succeeded.")
               print(f'Retrying... N. of trial:{count}')
               count += 1
               torch.cuda.empty_cache()
          except KeyboardInterrupt:
               print("Execution failed")
               break


def getListWeight(folderPath:str):
     """Return list of weights as PosixPath"""

     def sorter(x):
          # Helper
          x = x.stem.split('_')[-1]
          if x == 'latest':
               return 9999
          else:
               return int(x)
          
     ws = Path(folderPath).glob('*')
     ws = [x for x in ws if x.suffix=='.pth']
     ws.sort(key=sorter)
     return ws


def selector(selection):
     if selection == 'retina':
          config_path = cwd+"/MyConfigs/RetinaNet.py"

     if selection == 'mask50':
          config_path = cwd+"/MyConfigs/Mask_RCNN_r50.py"

     if selection == 'mask101':
          config_path = cwd+"/MyConfigs/Mask_RCNN_r101.py"

     if selection == 'cascade':
          config_path = cwd+"/MyConfigs/Cascade_Mask_RCNN.py"

     if selection == 'detr':
          config_path = cwd+"/MyConfigs/Detr_r50.py"

     if selection == 'centripetalnet':
          config_path = cwd+"/MyConfigs/centripetalnet.py"

     if selection == 'vfnet':
          config_path = cwd+"/MyConfigs/vfnet_r50.py"
     
     if selection == 'efficientdet':
          config_path = cwd+'/MyConfigs/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'
     
     if selection == 'sparce_rcnn':
          config_path = cwd+"/MyConfigs/sparce_rcnn.py"

     if selection == 'yolov3':
          config_path = cwd+"/MyConfigs/yolov3_d53.py"

     if selection == 'hrnet32':
          config_path = cwd+"/MyConfigs/HrNet_w32_cascade.py"

     if selection == 'hrnet40':
          config_path = cwd+"/MyConfigs/HrNet_w40_cascade.py"

     return config_path


def plotRes(workdir:str, title:str):
     json_files = Path(workdir).glob('**/*')
     json_files = [x for x in json_files if x.suffix == '.json']
     json_files = [x for x in json_files if x.name.startswith('e')]
     json_files=json_files.sort()
     Metrics = {'AP_50':[]}
     for idx in range(len(json_files)):
          df = pd.read_json(json_files[idx], encoding= 'unicode_escape', lines=True)
          metric = df['metric']
          metrics = metric.to_list()[0]
          AP50 = metrics['bbox_mAP_50']
          Metrics['AP_50'].append(AP50)
     plt.figure(dpi=300)
     plt.plot(Metrics['AP_50'])
     plt.xlabel('Epoch')
     plt.title(title +' ' + str(max(Metrics['AP_50'])))
     plt.ylabel('$mAP_{50}$')
     plt.savefig(f'{workdir}/{title}.png')
     plt.show()

def getCurrentTime():
     """Returns the current time as a string """
     now=datetime.now()
     now=now.isoformat()
     now=now.split(".")[0]
     return now