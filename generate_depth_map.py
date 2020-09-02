import argparse
import numpy as np
import tochvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from PIL.Image as Image

import os
import glob
import tqdm
# from calibration import RealsenseCalibration

import disp_models
import models
import logger

# ./src/generate_depth_map.py

class SubmitDataset(object):
    
    def __init__(self, file_path, dynamic_bs):
        self.dynamic_bs = dynamic_bs
        self.left_folder = 'image_2/'
        self.right_folder = 'image_3/'
        self.calib_folder = 'calib/'
        
        # file_path = './MyDataset/LIDAR/Data/'
        self.left_test = os.listdir(file_path + self.left_folder)
        self.right_test = os.listdir(file_path + self.right_folder)
        self.calib_test = os.listdir(file_path + self.calib_folder)
        
        # normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         normalize])
        
        
    def __getitem__(self, item):
        left_img = self.left_test[item]
        right_img = self.right_test[item]
        calib_info = self.calib_test[item]
        
        # ToDo: scale adjustment? => 1.0??
        calib = np.reshape(calib_info['P'], [3, 4])[0, 0] * 0.54
        
        imgL = Image.open(left_img).convert('RGB')
        imgR = Image.open(right_img).convert('RGB')
        
        # transformation of image
        imgL = self.trans(imgL)
        imgL = imgL[None, :, :, :]  # add batch dim
        imgR = self.trans(imgR)
        imgR = imgR[None, :, :, :]  # add batch dim
        
        B, C, H, W = imgL.shape
        
        # padding
        top_pad = 384 - H
        right_pad = 1248 - W
        
        imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        
        filename = self.left_test[item].split('/')[-1][:-4]
        
        # ToDo: may change calib => calib.item()
        return imgL[0].float(), imgR[0].float(), calib.item(), H, W, filename
    
    
    def __len__(self):
        
        return len(self.left_test)
    
    
    
def disp2depth(output, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    
    return depth
    
    
# inference
def inference(imgL_crop, imgR_crop, calib, model, data_type):
    # switch eval mode
    model.eval()
    imgL, imgR, calib = imgL_crop.cuda(), imgR_crop.cuda(), calib.cuda()
    
    with torch.no_grad():
        output = model(imgL, imgR, calib)
        
    if data_type == 'disparity':
        output = disp2depth(output, calib)
        
    pred_disp = output.data.cpu().numpy()  # to cpu and numpy
    
    return pred_disp


"""
command line
python ./src/generate_depth_map.py --dataset_path ./MyDataset/LIDAR/Data --resume ./results/sdn_kitti_train_set/sdn_kitti_object.pth 
--save_path ./results/sdn_kitti_train_set/ --data_tag Data

"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./MyDataset/LIDAR/Data/')
parser.add_argument('--resume', type=str, default='./results/sdn_kitti_train_set/sdn_kitti_object.pth')
parser.add_argument('--save_path', type=str, default='./results/sdn_kitti_train_set/')
parser.add_argument('--data_tag', type=str, default='Data')

args = parser.parse_args()

    
dataset_path = args.dataset_path
resume = args.resume  # pretrained model for sdnnet
save_path = args.save_path
data_tag = args.data_tag

dynamic_bs = False
batch_size = 4  # for test, batch_size = 4, for train, batch_size = 12
num_workers = 8

maxdisp = 192
maxdepth = 80
down = 2

arch = 'SDNet'  # 'SDNet' or 'PSMNet'

data_type = 'depth'  # 'depth' or 'disparity'

# get dataset loader
dataset = SubmitDataset(dataset_path, dynamic_bs)

# create image data loader
TestImageLoader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers, drop_last=False)

# load model
if data_type == 'disparity':
    model = disp_models.__dict__[arch](maxdisp=maxdisp)
elif data_type == 'depth':
    model = models.__dict__[arch](maxdepth=maxdepth, maxdisp=maxdisp, down=down)

model = nn.DataParallel(model).cuda()
torch.backends.cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.1)

# resume
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch']
optimizer.load_state_dict(checkpoint['optimizer'])
best_RMSE = checkpoint['RMSE']
scheduler.load_state_dict(checkpoint['scheduler'])

os.makedirs(save_path + '/depth_map/' + data_tag)

tqdm_eval_loader = tqdm(TestImageLoader, total=len(TestImageLoader))

for batch_idx, (imgL_crop, imgR_crop, calib, H, W, filename) in enumerate(tqdm_eval_loader):
    # inference
    pred_disp = inference(imgL_crop, imgR_crop, calib, model, data_type)
    # save
    for idx, name in enumerate(filename):
        np.save(save_path + '/depth_map/' + data_tag + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])


