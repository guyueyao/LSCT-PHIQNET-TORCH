
import torch
from torch.optim import Adam,SGD
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from scipy import stats

from LSCT import LSCT
from CVDVideo import CVDSource,CVDFeat
from ScheduleTool.Queue import QueueLine
if __name__ == "__main__":
    ql=QueueLine()
    ql.queueForRun()
    # SEED = 42
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #
    # torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda")

    konsource = CVDSource()
    for r in range(0,1):
        rounds_index = r
        checkpoint = konsource.fiveFolds[r]
        train_images = checkpoint['train_images']
        train_dmos = checkpoint['train_dmos_normed']
        test_images = checkpoint['test_images']
        test_dmos = checkpoint['test_dmos_normed']

        train_set = CVDFeat( images=train_images, dmos=train_dmos)
        test_set = CVDFeat( images=test_images, dmos=test_dmos)
        dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
        testloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

        model = LSCT().float().to(device)  #
        criterion = nn.MSELoss()  # L1 loss
        optimizer = Adam(model.parameters(), lr=1e-4)
        srocclast=0

        for epoch in range(400):
            # if epoch==100:
            #     optimizer=SGD(model.parameters(),lr=5e-5)
            # Train
            model.train()
            L = 0
            for idx, data in enumerate((dataloader)):
                features=data['feat']
                label=data['label']
                length=data['length']
                features = features.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  #
                outputs = model(features)
                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()
                # print(str(torch.mean(outputs).item()) + '   ' + str(loss.item()))
                L = L + loss.item()
            train_loss = L / (idx + 1)

            model.eval()
            pre = np.array([0])
            tar = np.array([0])
            for idx, data in enumerate((testloader)):
                features=data['feat']
                label=data['label']
                length=data['length']
                features = features.to(device).float()
                label = label.data.numpy().flatten()
                output = model(features)
                output = output.to('cpu')
                predict = output.data.numpy().flatten()
                pre = np.hstack((pre, predict))
                tar = np.hstack((tar, label))
            srocc1, _ = stats.spearmanr(pre[1:], tar[1:])
            plcc1, _ = stats.pearsonr(pre[1:], tar[1:])
            rmse1 = np.sqrt(np.mean(np.square(pre[1:] - tar[1:])))

            if srocc1>srocclast:
                srocclast=srocc1

                checkpoint={"model_state_dict": model.state_dict()}
                torch.save(checkpoint,'./cvdmodel/model_'+str(rounds_index)+'.pth')
            print('epoch %d , srocc %.4f ,srocc best %.4f' % (epoch, srocc1, srocclast))
    ql.deregister()
