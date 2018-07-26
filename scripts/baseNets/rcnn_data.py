# -*- coding: utf-8 -*-

import  os, sys
import  xml
import random

import torch
import torch.utils.data as tdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import selectivesearch as ss

import numpy as np
import cv2
import pickle
import xml.dom.minidom as xml_parse

class Data(tdata.Dataset):

    def __init__(self,data_path=None,model="cnn"):

        self.target=[0,1,2,3] # 0 is background, we just use three class of images
        self.catcher={}

        self.dataset_file=[]
        if data_path:
            with open(data_path,"r") as f:
                self.dataset_file = pickle.load(f)

        self.cnn_iou=0.5
        self.svm_iou=0.3
        self.regression_iou=0.6
        self.positive_num = 32  # svm and regression keep the same sample distribution
        self.negative_num = 96
        self.model = model
        self.positive_set=[]
        self.negative_set=[]

        self.length = 0
        self.positive_counter=0
        self.negative_counter=0
        self.positive_index=0
        self.negative_index=0

        threshold_iou=0

        if self.model=="cnn":
            threshold_iou=self.cnn_iou
        elif self.model=="svm":
            threshold_iou=self.svm_iou
        else:
            threshold_iou=self.regression_iou

        if len(self.dataset_file)!=0:
            for sample in self.dataset_file:
                if sample["iou"] < threshold_iou:
                    self.negative_set.append(sample)
                else:
                    self.positive_set.append(sample)

            plength=len(self.positive_set)
            nlength=len(self.negative_set)
            if plength * 3 >= nlength:
                self.length = nlength / 3 + nlength
            else:
                self.length=plength * 4

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        sample=None
        rand=random.randint(0,1)

        if self.positive_index == len(self.positive_set):
            self.positive_index = 0
        if self.negative_index == len(self.negative_set):
            self.negative_index = 0

        if self.positive_counter==self.positive_num and self.negative_counter==self.negative_num:
            self.positive_counter=0
            self.negative_counter=0

        if rand==0:
            if self.positive_counter<self.positive_num:
                sample=self.positive_set[self.positive_index]
                self.positive_counter+=1
            else:
                sample = self.negative_set[self.negative_index]
                self.negative_counter += 1
        else:
            if self.negative_counter<self.negative_num:
                sample=self.negative_set[self.negative_index]
                self.negative_counter+=1
            else:
                sample = self.positive_set[self.positive_index]
                self.positive_counter += 1

        img,pts,category,iou=sample[0],sample[1],sample[2],sample[3] # image, pts, category, iou
        if img in self.catcher:
            image=self.catcher[img][:,pts[0]:pts[0]+pts[2],pts[1]:pts[1]+pts[3]]
        else:
            image=cv2.imread(img)
            image=image/255
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

    def make_dataset(self, data_path=["n01798484","n02089973","n02110341"]):

        dt_root_dir="/workspace/dataset/imagenet/raw-data/train"
        bdx_root_dir="/workspace/dataset/imagenet/raw-data/bounding_boxes"
        count=0

        for i in range(len(data_path)):
            bds = os.listdir(bdx_root_dir+"/"+data_path[i])
            for bd in bds:
                img=dt_root_dir+"/"+data_path[i]+"/"+bd.split(".")[0]+".JPEG"

                info=xml_parse.parse(bdx_root_dir+"/"+data_path[i]+"/"+bd)
                root_tag=info.documentElement
                bndbox=root_tag.getElementsByTagName("bndbox")
                if len(bndbox)==0:
                    print("no valid boundingbox, image: ",img)
                    continue

                ox0=int(bndbox[0].getElementsByTagName("xmin")[0].childNodes[0].data)
                oy0=int(bndbox[0].getElementsByTagName("ymin")[0].childNodes[0].data)
                ox1=int(bndbox[0].getElementsByTagName("xmax")[0].childNodes[0].data)
                oy1=int(bndbox[0].getElementsByTagName("ymax")[0].childNodes[0].data)

                image=cv2.imread(img)
                img_lb, regions = ss.selective_search(image, scale=200, sigma=0.8, min_size=50)

                one = (img, (ox0,oy0,ox1-ox0,oy1-oy0), i + 1, 1.0)
                self.dataset_file.append(one)
                count=count+1

                rect_collectons=set()
                for reg in regions:
                    if reg["size"]<=50 or reg["rect"] in rect_collectons:
                        continue
                    rect_collectons.add(reg["rect"])
                    print("---", count, "---one: ", one)
                    iou=self.IOU((ox0,oy0,ox1-ox0,oy1-oy0),reg["rect"])
                    one=(img,reg["rect"],i+1,iou) # 0 is background
                    self.dataset_file.append(one)
        self.dataset_file=sorted(self.dataset_file,key=lambda dt:dt[3])
        self.save_dataset()

    def save_dataset(self,path="dataset_file.pkl"):
        print("samples num:%v",len(self.dataset_file))
        with open(path,"wb") as f:
            pickle.dump(self.dataset_file,f)

    def load_dataset(self,path="dataset_file.pkl"):
        print("dataset path: ",path)
        with open(path,"r") as f:
            self.dataset_file=pickle.load(f)
        print("dataset type: ",type(self.dataset_file),", dataset size: ",len(self.dataset_file))

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

if __name__=="__main__":

    dt=Data()
    dt.make_dataset()