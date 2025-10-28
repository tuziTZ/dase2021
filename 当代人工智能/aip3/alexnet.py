import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AlexNet, self).__init__()

        # 第一层卷积层
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二层卷积层
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # # 第三层卷积层
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        #
        # # 第四层卷积层
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        # 第四层卷积层
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        # 第五层卷积层
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2304, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加dropout层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)  # 添加dropout层
        )
        # 输出层
        self.output_layer = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        # x = x.view(-1, 256*6*6)
        # x = torch.flatten(x, 1)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.output_layer(x)
        return x
