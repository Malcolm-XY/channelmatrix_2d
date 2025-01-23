# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D_3layers_avgpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN2D_3layers_avgpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=4, stride=4)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class CNN2DModel_3layers_maxpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN2DModel_3layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
class CNN2D_3layers_maxpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN2D_3layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化层
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化层
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + BatchNorm + 池化层
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二层卷积 + 激活 + 池化
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三层卷积 + 激活 + 池化
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 全局池化
        x = self.global_pool(x)

        # 展平层
        x = x.view(x.size(0), -1)

        # 全连接层 1 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 全连接层 2 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 输出层
        x = self.fc3(x)

        return x
    
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 分支1
        branch1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        branch1 = self.branch1_pool1(branch1)
        branch1 = F.relu(self.branch1_bn2(self.branch1_conv2(branch1)))
        branch1 = self.branch1_pool2(branch1)

        # 分支2
        branch2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        branch2 = self.branch2_pool1(branch2)
        branch2 = F.relu(self.branch2_bn2(self.branch2_conv2(branch2)))
        branch2 = self.branch2_pool2(branch2)

        # 分支3
        branch3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        branch3 = self.branch3_pool1(branch3)
        branch3 = F.relu(self.branch3_bn2(self.branch3_conv2(branch3)))
        branch3 = self.branch3_pool2(branch3)

        # 融合分支
        branch1 = self.global_pool(branch1).view(x.size(0), -1)
        branch2 = self.global_pool(branch2).view(x.size(0), -1)
        branch3 = self.global_pool(branch3).view(x.size(0), -1)

        # 拼接特征
        combined = torch.cat([branch1, branch2, branch3], dim=1)

        # 全连接层
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)

        return x
    
class MultiScaleCNN_(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch1_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 新增

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2_conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)  # 新增

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch3_conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)  # 新增

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 分支1
        branch1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        branch1 = self.branch1_pool1(branch1)
        branch1 = F.relu(self.branch1_bn2(self.branch1_conv2(branch1)))
        branch1 = self.branch1_pool2(branch1)
        branch1 = F.relu(self.branch1_conv3(branch1))  # 新增卷积层

        # 分支2
        branch2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        branch2 = self.branch2_pool1(branch2)
        branch2 = F.relu(self.branch2_bn2(self.branch2_conv2(branch2)))
        branch2 = self.branch2_pool2(branch2)
        branch2 = F.relu(self.branch2_conv3(branch2))  # 新增卷积层

        # 分支3
        branch3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        branch3 = self.branch3_pool1(branch3)
        branch3 = F.relu(self.branch3_bn2(self.branch3_conv2(branch3)))
        branch3 = self.branch3_pool2(branch3)
        branch3 = F.relu(self.branch3_conv3(branch3))  # 新增卷积层

        # 融合分支
        branch1 = self.global_pool(branch1).view(x.size(0), -1)
        branch2 = self.global_pool(branch2).view(x.size(0), -1)
        branch3 = self.global_pool(branch3).view(x.size(0), -1)

        # 拼接特征
        combined = torch.cat([branch1, branch2, branch3], dim=1)

        # 全连接层
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)

        return x
