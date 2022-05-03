# Config my baseline with 
import os
import torch
torch.cuda.empty_cache()

os.system('clear')
cwd = os.getcwd()

def test(selection:str,checkpoint, workdir):
     if selection == 'retina':
          config_path = cwd+"/configs/MyBaselines/RetinaNet.py"

     if selection == 'mask50':
          config_path = cwd+"/configs/MyBaselines/Mask_RCNN_r50.py"

     if selection == 'mask101':
          config_path = cwd+"/configs/MyBaselines/Mask_RCNN_r101.py"

     if selection == 'cascade':
          config_path = cwd+"/configs/MyBaselines/Cascade_Mask_RCNN.py"

     if selection == 'detr':
          config_path = cwd+"/configs/MyBaselines/Detr_r50.py"

     if selection == 'centripetalnet':
          config_path = cwd+"/configs/MyBaselines/centripetalnet.py"

     if selection == 'vfnet':
          config_path = cwd+"/configs/MyBaselines/vfnet_r50.py"
     
     if selection == 'efficientdet':
          config_path = cwd+'/configs/MyBaselines/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'
     
     
     outfile = workdir +'/savings.pkl'
     eval_type = 'bbox'
     os.system(f'python tools/test.py {config_path} {checkpoint} --work-dir {workdir} --out {outfile} --eval {eval_type}')






def train(selection: str):
     if selection == 'retina':
          config_path = cwd+"/configs/MyBaselines/RetinaNet.py"

     if selection == 'mask50':
          config_path = cwd+"/configs/MyBaselines/Mask_RCNN_r50.py"

     if selection == 'mask101':
          config_path = cwd+"/configs/MyBaselines/Mask_RCNN_r101.py"

     if selection == 'cascade':
          config_path = cwd+"/configs/MyBaselines/Cascade_Mask_RCNN.py"

     if selection == 'detr':
          config_path = cwd+"/configs/MyBaselines/Detr_r50.py"

     if selection == 'centripetalnet':
          config_path = cwd+"/configs/MyBaselines/centripetalnet.py"

     if selection == 'vfnet':
          config_path = cwd+"/configs/MyBaselines/vfnet_r50.py"
     
     if selection == 'efficientdet':
          config_path = cwd+'/configs/MyBaselines/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'
     # os.system(f'bash tools/dist_train.sh {config_path} 1')
     
     MAX_TRIES = 1 # max number of training retrials
     
     count = 0
     while count < MAX_TRIES:
          try:
               # os.system(f'python tools/train.py {config_path} --auto-resume --seed 0')
               os.system(f'bash tools/dist_train.sh {config_path} 1 --auto-resume --seed 0')
               print("Execution succeeded.")
               print(f'Retrying... N. of trial:{count}')
               count += 1
               torch.cuda.empty_cache()
          except KeyboardInterrupt:
               print("Execution failed")
               break


if __name__ == '__main__':


     MODE = 'train'
     
     if MODE == 'train':
          selection = 'detr'
          print(f'Starting training of {selection}!')
          train(selection=selection)

     elif MODE == 'test':
          selection = 'detr'
          print(f'Starting evaluating of {selection}!')
          checkpoint = "/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/checkpoints/DETR_r50_150e/epoch_150.pth"
          workdir = "/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/checkpoints/out"
          test(selection=selection, checkpoint=checkpoint, workdir=workdir)
