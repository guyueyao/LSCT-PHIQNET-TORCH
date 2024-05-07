import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from KonData import KonData_pro,KonTest_pro,KonSource

from phiqnet import PHIQNet
torch.backends.cudnn.benchmark = True
batch_size = 8
device = torch.device('cuda')

net=PHIQNet()
net = net.half().to(device)
criterion = nn.MSELoss()
Opt = torch.optim.Adam(net.parameters(), lr=5e-5, eps=1e-5)
soureData=KonSource()
train_images, train_dmos, test_images, test_dmos=soureData.train_test()
trainset = KonData_pro(root_dir='/home1/server823-2/database/Kon10k/koniq10k_1024x768/Images/', images=train_images, dmos=train_dmos)
testset = KonTest_pro(root_dir='/home1/server823-2/database/Kon10k/koniq10k_1024x768/Images/',images=test_images, dmos=test_dmos)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=8)
testloader = DataLoader(testset, batch_size=8, shuffle=True,num_workers=8)

trainset2 = KonData_pro(root_dir='/home2/512x384/', images=train_images, dmos=train_dmos)
testset2 = KonTest_pro(root_dir='/home2/512x384/',images=test_images, dmos=test_dmos)
dataloader2 = DataLoader(trainset2, batch_size=batch_size, shuffle=True,num_workers=8)
testloader2 = DataLoader(testset2, batch_size=8, shuffle=True,num_workers=8)
running_loss = 0
batc = 0
start_epoch=0
srocclast=0
me = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to(
            'cuda')
st = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to(
            'cuda')
bat=0
for epoch in range(start_epoch, start_epoch + 50):
    net.train()

    for loader in [dataloader]:
        for idx, data in enumerate(tqdm(dataloader)):
            inputs = data['image']
            labels = data['label']
            inputs, labels = inputs.half(), labels.half().view(-1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                inputs=inputs/255.0
                inputs=inputs.sub(me).div(st)

            output = net(inputs).view(-1)
            loss = criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            bat=bat+1
            if bat>3:
                Opt.step()
                Opt.zero_grad()
                bat=0
            # lr_scheduler.step()
    running_loss = running_loss / len(trainset)/2

    with torch.no_grad():
        net.eval()
        pre = np.array([0])
        tar = np.array([0])
        for loader in [testloader]:
            for idx, data in enumerate(tqdm(testloader)):
                inputs = data['image']
                labels = data['label']
                # inputs = inputs[:,:, 0::2, :, :]
                inputs, labels = inputs.half(), labels.data.numpy().flatten()
                inputs = inputs.to(device)
                with torch.no_grad():
                    inputs=inputs/255.0
                    inputs=inputs.sub(me).div(st)

                output = net(inputs)
                output = output.to('cpu')
                predict = output.data.numpy().flatten()
                pre = np.hstack((pre, predict))
                tar = np.hstack((tar, labels))
        srocc, _ = stats.spearmanr(pre[1:], tar[1:])
        print('epc: ' + str(epoch) + '   loss: ' + str(running_loss) + '    srocc: ' + str(srocc))
        running_loss = 0
        if (True):
            checkpoint = {"model_state_dict": net.state_dict(),
                          "epoch": epoch,
                          "train_images": train_images,
                          "train_dmos": train_dmos,
                          "optimizer_state_dict": Opt.state_dict(),
                          "test_images": test_images,
                          "test_dmos": test_dmos}
            path_checkpoint = "./model10K/phiqnet_%d_%d.pth"%(epoch,int(srocc*100))
            torch.save(checkpoint, path_checkpoint)
            if srocc > srocclast:
                path_checkpoint = "./model10K/phiqnetv2.pth"
                torch.save(checkpoint, path_checkpoint)
                srocclast = srocc
