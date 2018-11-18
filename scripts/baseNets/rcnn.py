import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as udata

import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import linear_model
import selectivesearch as ss
import json

import rcnn_data


transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.ToPILImage(),transforms.Resize((227,227)),transforms.ToTensor()])

class CRNN(nn.Module):

    def __init__(self,num_classes):

        super(CRNN,self).__init__()

        alexnet = models.alexnet(pretrained=True)

        self.Tclassifier=False

        self.svm_classifier=svm.SVC(kernel="linear",probability=True)
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

    def forward(self,img):

        x=self.features(img)
        x=x.view(-1,256*6*6)
        x=self.flatten(x)

        if self.Tclassifier:
            return x

        if not self.training:
            ## if not in training model, add svm at the end of flatten
            x=x.tolist()
            clip=self.ridge_regression.predict(x)
            clas=self.svm_classifier.predict_proba(x)
            x={"clip":clip,"class":clas}
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
    net.load(model="classifier")

    train_svm_set = rcnn_data.Data(data_path="dataset_file_390459_samples.pkl", model="svm", transfomer=transf)
    dataloader = udata.DataLoader(train_svm_set,batch_size=128,shuffle=False)

    features=[]
    svm_targets=[]
    for i,data in enumerate(dataloader,0):
        inputs, labels, _ = data["image"], data["target"],data["groundtruth"]
        inputs, labels=inputs.to(device),labels.to(device)
        outputs=net(inputs)
        features.extend(outputs.tolist())
        svm_targets.extend(labels.tolist())

    net.svm_classifier.fit(features, svm_targets)

    train_svm_set = rcnn_data.Data(data_path="dataset_file_390459_samples.pkl", model="ridge", transfomer=transf)
    dataloader = udata.DataLoader(train_svm_set, batch_size=128, shuffle=False)
    features=[]
    ridge_target=[]
    for i,data in enumerate(dataloader,0):
        inputs, pts, groundtruth = data["image"], data["pts"],data["groundtruth"]
        inputs, labels=inputs.to(device),labels.to(device)
        outputs=net(inputs)
        features.extend(outputs.tolist())
        ridge_target.extend((pts[0]-groundtruth[0],pts[1]-groundtruth[1],pts[2]/groundtruth[2],pts[3]/groundtruth[3]))

    print("ridge regression sample num:", len(ridge_target))
    net.ridge_regression.fit(features,ridge_target)
    net.saver(model="classifier")


def Inference(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    objclass=4
    net = CRNN(num_classes=objclass)
    net.to(device)
    net.load(model="all")
    net.eval()

    _, regions = rcnn_data.Data.rect_select(image)

    clips=[]
    classes = []
    ptss= []
    for reg in regions:
        pts=reg["rect"]
        ptss.append(pts)
        image=transf(image[pts[1]:pts[1] + pts[3],pts[0]:pts[0] + pts[2],:])
        output=net(image)
        clips.append(output["clip"])
        classes.append(output["class"])

    stclass=np.argsort(classes)
    class_rects=[]
    for i in xrange(len(classes)):
        stclass[i].reverse()
        ret_cl=rcnn_data.Data.nms(ptss[stclass[i]],clips[stclass[i]])
        class_rects.append(ret_cl)
        print(">>>>>>>>>>>>>>>>>>>>>>>> class: ",i, "result pts:",ret_cl)

    result_file=open("pts_result.file")
    json.dump(result_file,class_rects)
    result_file.close()
    ##canny检测这一步不做，直接bridge regression应用
    #每一个图片的所有候选框n送入svm，得到n*4个打分，每列为一个类别对所有框的打分，对非背景的3列做NMS
    #最后设定一个剔除阈值来剔除所有不合格的框，这样也会把图片中不可能存在的那类物体的框全部剔除从而只留下
    #最可能存在的类别中的最可能的框



if __name__=="__main__":
    TrainSVMandRidgeRegression()






