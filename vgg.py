# Defining VGG
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['relu11'] = F.relu(self.conv1_1(x))
        out['relu12'] = F.relu(self.conv1_2(out['relu11']))
        out['pool1'] = self.pool1(out['relu12'])
        out['relu21'] = F.relu(self.conv2_1(out['pool1']))
        out['relu22'] = F.relu(self.conv2_2(out['relu21']))
        out['pool2'] = self.pool2(out['relu22'])
        out['relu31'] = F.relu(self.conv3_1(out['pool2']))
        out['relu32'] = F.relu(self.conv3_2(out['relu31']))
        out['relu33'] = F.relu(self.conv3_3(out['relu32']))
        out['relu34'] = F.relu(self.conv3_4(out['relu33']))
        out['pool3'] = self.pool3(out['relu34'])
        out['relu41'] = F.relu(self.conv4_1(out['pool3']))
        out['relu42'] = F.relu(self.conv4_2(out['relu41']))
        out['relu43'] = F.relu(self.conv4_3(out['relu42']))
        out['relu44'] = F.relu(self.conv4_4(out['relu43']))
        out['pool4'] = self.pool4(out['relu44'])
        out['relu51'] = F.relu(self.conv5_1(out['pool4']))
        out['relu52'] = F.relu(self.conv5_2(out['relu51']))
        out['relu53'] = F.relu(self.conv5_3(out['relu52']))
        out['relu54'] = F.relu(self.conv5_4(out['relu53']))
        out['pool5'] = self.pool5(out['relu54'])
        
        return [out[key] for key in out_keys]