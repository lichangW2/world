import  os, sys
import  xml

import torch
import torch.utils.data as tdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import selectivesearch as ss

import  numpy as np
import  cv2
import pickle
from xml.dom.minidom import parse as xml_parse

class Data(tdata.Dataset):

    def __init__(self,data_path=None):

        self.target=[0,1,2,3] # 0 is background, we just use three class of images
        self.catcher={}

        self.dataset_file=[]
        if not data_path:
            self.dataset_file = pickle.load(data_path)

        self.cnn_iou=0.5
        self.svm_iou=0.3
        self.model="cnn"

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, item):

        img,pts,category,iou=self.dataset_file[item] # image, pts, category, iou
        if img in self.catcher:
            image=self.catcher[img][:,pts[0]:pts[0]+pts[2],pts[1]:pts[1]+pts[3]]
        else:
            image=cv2.imread(img)
            self.catcher[img]=image.copy()
            image = self.catcher[img][:, pts[0]:pts[0] + pts[2], pts[1]:pts[1] + pts[3]]

        if len(self.catcher)>=30:
            self.catcher={}

        iou_threshold=self.cnn_iou
        if self.model!="cnn":
            iou_threshold=self.svm_iou

        target=0
        if iou >=iou_threshold:
            target=category

        return {"image":image,"target":target}

    def set_train_model(self, model="cnn"):
        self.model=model

    def make_dataset(self, data_path=["n01798484","n02089973","n02110341"],save_path="make_datasets"):

        dt_root_dir="/disk1/dataset/imagenet/raw-data/train"
        bdx_root_dir="/disk1/dataset/imagenet/raw-data/bounding_boxes"

        for i in xrange(len(data_path)):
            bds = os.listdir(bdx_root_dir+"/"+data_path[i])
            for bd in bds:
                img=dt_root_dir+"/"+data_path[i]+"/"+bd.split(".")[0]+".JPEG"

                info=xml_parse.parse(bdx_root_dir+"/"+data_path[i]+"/"+bd)
                root_tag=info.documentElement
                bndbox=root_tag.getElementsByTagName("bndbox")
                if len(bndbox)==0:
                    print("no valid boundingbox, image: ",img)
                    continue

                ox0=int(bndbox[0].getElementsByTagName("xmin").childNodes[0].data)
                oy0=int(bndbox[0].getElementsByTagName("ymin").childNodes[0].data)
                ox1=int(bndbox[0].getElementsByTagName("xmax").childNodes[0].data)
                oy1=int(bndbox[0].getElementsByTagName("ymax").childNodes[0].data)

                image=cv2.imread(img)
                img_lb, regions = ss.selective_search(image, scale=1000, sigma=0.8, min_size=50)

                one = (img, (ox0,oy0,ox1-ox0,oy1-oy1), i + 1, 1.0)
                self.dataset_file.append(one)
                for reg in regions:
                    if reg["size"]<=50:
                        continue
                    iou=self.IOU((ox0,oy0,ox1-ox0,oy1-oy1),reg["rect"])
                    one=(img,reg["rect"],i+1,iou) # 0 is background
                    self.dataset_file.append(one)
        self.dataset_file=sorted(self.dataset_file,key=lambda dt:dt[3])
        pickle.dump(self.dataset_file,save_path)

    def save_dataset(self,path="dataset_file.pkl"):
        pickle.dump(self.dataset_file,path)

    def IOU(self,rect1,rect2):
        # iou=0.5 训练alexnet
        # iou=0.3 训练


        r1x1=rect1[0]
        r1y1=rect1[1]
        r1x2=r1x1+rect1[2]
        r1y2=r1y1+rect1[3]

        r2x1=rect2[0]
        r2y1=rect2[1]
        r2x2=r2x1+rect2[2]
        r2y2=r2y1+rect2[3]

        r1area=rect1[2]*rect1[3]
        r2area=rect2[2]*rect2[3]

        x1=max(r1x1,r2x1)
        y1=max(r1y1,r2y1)
        x2=min(r1x2,r2x2)
        y2=min(r1y2,r2y2)

        width=max(0,x2-x1)
        height=max(0,y2-y1)

        area=width*height

        iou=area/(r1area+r2area-area)
        return iou