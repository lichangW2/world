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
from sklearn import linear_model

import  rcnn_data


transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.ToPILImage(),transforms.Resize((227,227)),transforms.ToTensor()])

class CRNN(nn.Module):

    def __init__(self,num_classes):

        super(CRNN,self).__init__()

        alexnet = models.alexnet(pretrained=True)

        self.Tclassifier=False

        self.svm_classifier=svm.SVC(kernel="linear")
        self.ridge_regression=linear_model.Ridge(alpha=num_classes) ## input: (4096 features),output:(scale_x,scale_y,offset_x,offset_y)
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

        if self.Tclassifier:
            return x

        if not self.training:
            ## if not in training model, add svm at the end of flatten
            x=x.tolist()
            x=self.svm_classifier.predict(x)
        else:
            x= self.nn_classifier(x)

        return x


    def training_svm_classfier(self,train_svm=True):
        self.Tclassifier=train_svm


    def saver(self,path=".",model="cnn"):

        if model=="cnn":
            torch.save({"model": self.state_dict()}, path + "/crnn_feature.tar")
        elif model=="classifier":
            joblib.dump(self.classifier, path + "/crnn_classifier.tar")
            joblib.dump(self.ridge_regression,path + "/ridge_regression.tar")
        else:
            joblib.dump(self.classifier,path+"/crnn_classifier.tar")
            torch.save({"model":self.state_dict()},path+"/crnn_feature.tar")
            joblib.dump(self.ridge_regression, path + "/ridge_regression.tar")
        return

    def load(self,path=".",model="cnn"):

        if model=="cnn":
            self.load_state_dict(torch.load(path+"/"+"crnn_classifier.tar"))
        elif model=="classifier":
            self.classifier=joblib.load(path+"/crnn_classifier.tar")
            self.ridge_regression=joblib.load(path + "/ridge_regression.tar")
        else:
            self.load_state_dict(torch.load(path + "/" + "crnn_classifier.tar"))
            self.classifier = joblib.load(path + "/crnn_classifier.tar")
            self.ridge_regression = joblib.load(path + "/ridge_regression.tar")
        return


def TrainingCnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CRNN(num_classes=4)
    net.to(device)
    net.train()

    criterion = nn.MultiMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_set=rcnn_data.Data(data_path="dataset_file_390459_samples.pkl",model="cnn",transfomer=transf)
    dataloader = udata.DataLoader(train_set,batch_size=128,shuffle=False)

    print("dataset length: ",len(train_set))
    print("net state, traing: ", net.training)
    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(dataloader,0):
            inputs,labels =data["image"],data["target"]
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

def TrainSVMandRidgeRegression():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CRNN(num_classes=4)
    net.to(device)
    net.Tclassifier=True
    net.eval()
    net.load(model="classifier")

    train_set = rcnn_data.Data(data_path="dataset_file_390459_samples.pkl", model="svm", transfomer=transf)
    dataloader = udata.DataLoader(train_set,batch_size=128,shuffle=False)

    features=[]
    svm_targets=[]
    ridge_target=[]
    for i,data in enumerate(dataloader,0):
        inputs, labels = data["image"], data["target"],data["groundtruth"]
        inputs, labels=inputs.to(device),labels.to(device)
        outputs=net(inputs)
        features.extend(outputs.tolist())
        svm_targets.extend(labels.tolist())

    net.svm_classifier.fit(features, svm_targets)
    net.ridge_regression.fit()


def Inference():
    pass


if __name__=="__main__":

    TrainingCnn()






