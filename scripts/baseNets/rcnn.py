import torch
import torch.nn as nn
import  torchvision.models as models
import cv2
from sklearn import svm


import selectivesearch as ss

img=cv2.imread("/Users/cj/Desktop/db67975231824f17b6c376759237b3ee_w640_h360.vsample_000002.jpg")

img_lb,regions=ss.selective_search(img,scale=1000,sigma=0.8,min_size=50)

alexnet.classifier.modules()
print(alexnet)

#nn.MultiMarginLoss

print alexnet.classifier[6]

for parm in alexnet.classifier[6].parameters():
    print type(parm), parm.shape, parm


class CRNN(nn.Module):

    def __init__(self):

        super(CRNN,self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.features=alexnet.features
        self.flatten=nn.Sequential(
            alexnet.classifier[0],
            alexnet.classifier[1],
            alexnet.classifier[2],
            alexnet.classifier[3],
            alexnet.classifier[4]
        )
    def forward(self,x):
        if not self.training:
            ## if not in training model, add svm at the end of flatten

        else:

    def saver(self):
        pass
    def load(self):
        pass
