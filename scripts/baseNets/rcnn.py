import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as udata

import cv2
from sklearn import svm
from sklearn.externals import joblib

import  rcnn_data

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


        if not self.training:
            ## if not in training model, add svm at the end of flatten
            x=x.tolist()
            x=self.svm_classifier.predict(x)
        elif self.training:
            x= self.nn_classifier(x)

        return x


    def training_svm_classfier(self,train_svm=True):
        self.Tclassifier=train_svm


    def saver(self,path=".",model="cnn"):

        if model=="cnn":
            torch.save({"model": self.state_dict()}, path + "/crnn_feature.tar")
        elif model=="classifier":
            joblib.dump(self.classifier, path + "/crnn_classifier.tar")
        else:
            joblib.dump(self.classifier,path+"/crnn_classifier.tar")
            torch.save({"model":self.state_dict()},path+"/crnn_feature.tar")
        return

    def load(self,path):

        self.load_state_dict(torch.load(path+"/"+"crnn_classifier.tar"))
        self.classifier=joblib.load(path+"/crnn_classifier.tar")


def TrainingCnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CRNN()
    net.to(device)

    criterion = nn.MultiMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), transforms.Resize((227,227))])
    train_set=rcnn_data.Data()
    train_set.set_train_model()
    dataloader = udata.DataLoader(train_set,transformer,batch_size=128,shuffle=False)

    print("net state, traing: ", net.training)
    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(dataloader,0):
            inputs,labels =data
            inputs,labels=inputs.to(device),labels.to(device)

            net.zero_grad()
            outputs=net(inputs)

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if i%10==0:
                print("[%d, %d] loss: %0.3f"%(epoch+1,i+1,running_loss/10))
                running_loss=0.0

    print("Finished Training")
    net.saver()




if __name__=="__main__":








