import torch
import torch.nn as nn
from PHIQNET.myresnet_tf import resnet50
import torch.nn.functional as F

class Conv2dWithName(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,
                     padding=0, groups=1, use_bias=True, dilation=1,name=None):
        super(Conv2dWithName, self).__init__()
        self.conv2d=nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=use_bias, dilation=dilation)
        self.name=name
        self.use_bias=use_bias
    def forward(self,x):
        return self.conv2d(x)
    def set_weight(self,layer):
        with torch.no_grad():
            print('INFO: init layer %s with tf weights'%self.name)
            weights=layer.get_weights()
            weight=weights[0]
            weight=torch.from_numpy(weight)
            weight=weight.permute((3,2,0,1))
            self.conv2d.weight.copy_(weight)
            if self.use_bias:
                bias=weights[1]
                bias = torch.from_numpy(bias)
                self.conv2d.bias.copy_(bias)


class BatchNorm2dWithName(nn.Module):
    def __init__(self,n_chaanels,name=None):
        super(BatchNorm2dWithName, self).__init__()
        self.bn=nn.BatchNorm2d(n_chaanels)
        self.name=name
    def forward(self,x):
        return self.bn(x)

    def set_weight(self,layer):
        with torch.no_grad():
            print('INFO: init layer %s with tf weights' % self.name)
            weights=layer.get_weights()
            gamma=torch.from_numpy(weights[0])
            beta=torch.from_numpy(weights[1])
            run_mean=torch.from_numpy(weights[2])
            run_var= torch.from_numpy(weights[3])
            self.bn.bias.copy_(beta)
            self.bn.running_mean.copy_(run_mean)
            self.bn.running_var.copy_(run_var)
            self.bn.weight.copy_(gamma)
            self.bn.momentum=1-layer.momentum

class DenseWithName(nn.Module):
    def __init__(self,in_dim,out_dim,name=None):
        super(DenseWithName, self).__init__()
        self.dense=nn.Linear(in_dim,out_dim)
        self.name=name
    def set_weight(self,layer):
        print('INFO: init layer %s with tf weights' % self.name)
        with torch.no_grad():
            weights = layer.get_weights()
            weight = torch.from_numpy(weights[0]).transpose(0, 1)
            self.dense.weight.copy_(weight)
            bias = weights[1]
            bias = torch.from_numpy(bias)
            self.dense.bias.copy_(bias)
    def forward(self,x):
        return self.dense(x)


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
    def forward(self,source, target):
        target_shape = target.shape
        output=F.interpolate(source,size=(target_shape[-2],target_shape[-1]),mode='nearest')
        return output
class FPN(nn.Module):
    def __init__(self,name='fpn_'):
        super(FPN, self).__init__()
        self.C5_reduce_Layer=Conv2dWithName(2048,256,1,1,0,name=name + 'C5_reduced')
        self.upsample_Layer=Upsample()
        self.P5_Layer=Conv2dWithName(256,256,3,1,1,name=name + 'P5')

        self.C4_reduce_Layer = Conv2dWithName(1024, 256, 1, 1, 0,name=name + 'C4_reduced')
        self.P4_Layer =Conv2dWithName(256, 256, 3, 1, 1,name=name + 'P4')

        self.C3_reduce_Layer = Conv2dWithName(512, 256, 1, 1, 0,name=name + 'C3_reduced')
        self.P3_Layer =Conv2dWithName(256, 256, 3, 1, 1,name=name + 'P3')

        self.C2_reduce_Layer = Conv2dWithName(256, 256, 1, 1, 0,name=name + 'C2_reduced')
        self.P2_Layer = Conv2dWithName(256, 256, 3, 1, 1,name=name + 'P2')

        self.P6_Layer= Conv2dWithName(2048, 256, 3, 2, 1,name=name + 'P6')

    def forward(self,C2, C3, C4, C5):
        P5=self.C5_reduce_Layer(C5)
        P5_upsampled=self.upsample_Layer(P5,C4)
        P5=self.P5_Layer(P5)

        P4=self.C4_reduce_Layer(C4)
        P4=P5_upsampled+P4
        P4_upsampled=self.upsample_Layer(P4,C3)
        P4=self.P4_Layer(P4)

        P3=self.C3_reduce_Layer(C3)
        P3=P4_upsampled+P3
        P3_upsampled=self.upsample_Layer(P3,C2)
        P3=self.P3_Layer(P3)

        P2=self.C2_reduce_Layer(C2)
        P2=P3_upsampled+P2
        P2=self.P2_Layer(P2)

        P6=self.P6_Layer(C5)

        return P2, P3, P4, P5, P6

class Channel_Spatial_Attention(nn.Module):
    def __init__(self,n_channels,name):
        super(Channel_Spatial_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_dense_layer=nn.Sequential(DenseWithName(n_channels,n_channels,name='dense'+name),nn.Sigmoid())
        self.conv2d = nn.Sequential(Conv2dWithName(in_planes=2, out_planes=1, kernel_size=7, stride=1, padding=3,use_bias=False,name='conv2d'+name),nn.Sigmoid())
        self.dense=nn.Linear(n_channels,1)
    def forward(self,input_feature):
        channel_input = input_feature
        spatial_input = input_feature
        avgout = self.shared_dense_layer(self.avg_pool(channel_input).view(channel_input.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_dense_layer(self.max_pool(channel_input).view(channel_input.size(0), -1)).unsqueeze(2).unsqueeze(3)
        channel_weights=(avgout+maxout)/2
        avg_pool_spatial=torch.mean(spatial_input,dim=1,keepdim=True)
        max_pool_spatial,_=torch.max(spatial_input,dim=1,keepdim=True)
        spatial_weights=torch.cat([avg_pool_spatial,max_pool_spatial],dim=1)
        spatial_weights=self.conv2d(spatial_weights)
        outputs= input_feature*channel_weights*spatial_weights
        outputs=self.avg_pool(outputs)
        outputs=outputs.squeeze(-1).squeeze(-1)
        # outputs=self.dense(outputs)
        return outputs
class PHIQNet(nn.Module):
    def __init__(self):
        super(PHIQNet, self).__init__()
        self.backbone_model=resnet50()
        self.fpn=FPN()
        self.attention1=Channel_Spatial_Attention(256,name='')
        self.attention2 = Channel_Spatial_Attention(256,name='_1')
        self.attention3 = Channel_Spatial_Attention(256,name='_2')
        self.attention4 = Channel_Spatial_Attention(256,name='_3')
        self.attention5 = Channel_Spatial_Attention(256,name='_4')

    def load_weight_from_tf(self,tf_model):
        for m in self.modules():
            if isinstance(m, (Conv2dWithName,BatchNorm2dWithName,DenseWithName)):
                layer=tf_model.get_layer(m.name)
                m.set_weight(layer)
        self.backbone_model.init_from_tf(tf_model)

    def forward(self,x):
        C2,C3,C4,C5=self.backbone_model(x)
        P2, P3, P4, P5, P6=self.fpn(C2,C3,C4,C5)

        P2=self.attention1(P2)
        P3 = self.attention2(P3)
        P4 = self.attention3(P4)
        P5 = self.attention4(P5)
        P6 = self.attention5(P6)
        P=torch.cat([P2,P3,P4,P5,P6],dim=1)

        # outputs=torch.mean(P,dim=1)
        return P

