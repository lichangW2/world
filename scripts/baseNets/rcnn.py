import torch
import torch.nn as nn
import  torchvision.models as models
import cv2
from sklearn import svm
from sklearn.externals import joblib

import selectivesearch as ss

img=cv2.imread("/Users/cj/Desktop/db67975231824f17b6c376759237b3ee_w640_h360.vsample_000002.jpg")

img_lb,regions=ss.selective_search(img,scale=1000,sigma=0.8,min_size=50)

alexnet.classifier.modules()
print(alexnet)

#nn.MultiMarginLoss

print alexnet.classifier[6]

for parm in alexnet.classifier[6].parameters():
    print type(parm), parm.shape, parm

sv=svm.SVC()
sv.fit()

class CRNN(nn.Module):

    def __init__(self,num_classes):

        super(CRNN,self).__init__()

        alexnet = models.alexnet(pretrained=True)

        self.Tclassifier=False

        self.svm_classifier=svm.SVC(kernel="linear")
        self.features=alexnet.features
        self.flatten=nn.Sequential(
            alexnet.classifier[0],
            alexnet.classifier[1],
            alexnet.classifier[2],
            alexnet.classifier[3],
            alexnet.classifier[4]
        )
        self.nn_classifier= nn.Linear(4096, num_classes)

    def forward(self,x):

        x=self.features(x)
        x=x.view(-1,256*6*6)
        x=self.flatten(x)


        if not self.training and not self.Tclassifier:
            ## if not in training model, add svm at the end of flatten
            x=x.tolist()
            x=self.svm_classifier.predict(x)
        elif self.training and not self.Tclassifier:
            x= self.nn_classifier(x)

        return x


    def training_svm_classfier(self,train_svm=True):
        self.Tclassifier=train_svm


    def saver(self,path="."):

        joblib.dump(self.classifier,path+"/crnn_classifier.tar")
        torch.save({"model":self.state_dict()},path+"/crnn_feature.tar")

    def load(self,path):

        self.load_state_dict(torch.load(path+"/"+"crnn_classifier.tar"))
        self.classifier=joblib.load(path+"/crnn_classifier.tar")





def __name__=="__main__":




