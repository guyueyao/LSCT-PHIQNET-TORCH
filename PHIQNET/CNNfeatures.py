"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
# 
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

import torch

from torch.utils.data import Dataset

import numpy as np
import random
from tqdm import tqdm
import DATAConfig
from KonVideo import KonSource,KonData
# from CSIQVideo import CSIQData,CSIQSource
# from CVDVideo import CVDTest,CVDSource
from torch.utils.data import DataLoader
from phiqnet import PHIQNet
from ScheduleTool.Queue import QueueLine


if __name__ == "__main__":
    ql=QueueLine()
    ql.queueForRun()
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.utils.backcompat.broadcast_warning.enabled = True

    device = torch.device("cuda")
    konsource = CVDSource()
    checkpoint = konsource.fiveFolds[0]
    train_images = checkpoint['train_images']
    test_images = checkpoint['test_images']
    train_dmos = checkpoint['train_dmos_normed']
    test_dmos = checkpoint['test_dmos_normed']
    train_set=KonData(root_dir=DATAConfig.DATAConfig().KONV_SERVER2_PATH,images=train_images,dmos=train_dmos)
    test_set=KonData(root_dir=DATAConfig.DATAConfig().KONV_SERVER2_PATH,images=test_images,dmos=test_dmos)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)
    frame_batch_size=8
    extractor = PHIQNet().half().to(device)
    extractor.eval()
    extractor.load_state_dict(torch.load("./PIHQNET.pth")['model_state_dict'])
    for loader in [dataloader,testloader]:
        for idx, data in enumerate(tqdm(loader)):
            inputs = data['image']
            labels = data['label']
            vname = data['vname']

            with torch.no_grad():
                me=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')
                st = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')
                inputs=inputs.squeeze(0).half()
                # inputs=inputs.div(127.5).sub(1.0)
                # inputs/=255.0
                # inputs=inputs.sub_(me).div_(st)
            video_length = inputs.shape[0]
            frame_start = 0
            frame_end = frame_start + frame_batch_size

            with torch.no_grad():
                output1 = torch.Tensor().to(device)
                while frame_end < video_length:
                    batch = inputs[frame_start:frame_end].to(device)
                    batch=batch.div(127.5).sub(1.0)
                    features_mean= extractor(batch)
                    output1 = torch.cat((output1, features_mean), 0)
                    frame_end += frame_batch_size
                    frame_start += frame_batch_size

                last_batch = inputs[frame_start:video_length].to(device)
                last_batch=last_batch.div(127.5).sub(1.0)
                features_mean = extractor(last_batch)
                output1 = torch.cat((output1, features_mean), 0)
                features=output1.to('cpu').numpy()
                clip_features = []
                clip_length = 16
                for j in range(features.shape[0] // clip_length):
                    clip_features.append(features[j * clip_length: (j + 1) * clip_length, :])
                clip_features = np.array(clip_features)
                vname = vname[0][:-4]
                vname = vname.split('/')[-1]
                np.save('/home2/hzy/phiqnet/feats/'+vname+'.npy',clip_features)
            del inputs,batch,output1,features_mean
            torch.cuda.empty_cache()
    ql.deregister()