import os
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from PIL import Image


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).reshape(x.shape[0], -1)
        features = self.fc_features(features)
        return features


if __name__ == '__main__':
    dataset_path = "/home/newdisk/zjy/datasets/PascalSentenceDataset/dataset/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg19 = VGGNet()
    vgg19.cuda()

    image_list = []
    print("开始准备数据！")
    for parent, dirnames, filenames in os.walk(dataset_path):
        dirnames.sort()
        for dir in dirnames:
            image_name_list = os.listdir(parent + dir)
            image_name_list.sort()
            image_list.append(
                np.stack(
                    np.array(Image.open(parent + dir + '/' + img).resize((224, 224)), dtype=np.float32).transpose(
                        (2, 0, 1)) for img in
                    image_name_list))
    print("数据准备完毕！")
    print("开始提取特征！")
    feature_list = np.array(
        [vgg19.forward(torch.from_numpy(images).cuda()).detach().cpu().numpy() for images in image_list])
    feature_list = np.concatenate(feature_list)
    np.save("/home/newdisk/zjy/projects/DSCMR-master/data/pascal_sentence/image_features.npy", feature_list)

    print("提取特征完成！")
