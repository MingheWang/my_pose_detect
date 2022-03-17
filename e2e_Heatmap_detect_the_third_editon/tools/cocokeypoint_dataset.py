import cv2
import os
import numpy as np
import torch
from tools.create_cocodataset import CocoDataset
from tools.transforms import GetTransforms
from tools.get_heatmap import Generate_heatmap

class CocoKeypoint(CocoDataset):
    def __init__(self,cfg,root,dataset_name):
        super().__init__(root,dataset_name)
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.get_heatmap = Generate_heatmap(cfg.num_joints,cfg.heatmap_size,cfg.base_sigma)
        self.get_transforms = GetTransforms(self.image_size)        # resize , totensor
        self.hw_ratio = cfg.hw_ratio
        self.num_joints = cfg.num_joints
        self.base_sigma = cfg.base_sigma
        self.with_scale_sigma = cfg.with_scale_sigma

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)            # list
        img_name = self.coco.imgs[img_id]["file_name"]
        img_path = os.path.join(self.root,self.dataset_name,img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img_h,img_w,_ = img.shape
        # 从该图片中筛选出有关键点的人体 anno
        annos = [anno for anno in annos if anno['iscrowd']==0 or len(anno["num_keypints"])>0]
        person_boxs = self.adjust_box(annos,img_h,img_w)    # numpy.shape=(num_person,4)  (x1,y1,w,h)
        joints = self.get_joints(annos)     # shape=(n_person,17,4) 或 (n_person,17,3)  $ 4 = x,y,vix,sigma
        img,person_boxs,joints = self.transform_fn(img,person_boxs,joints)
        # 节点相对人体框的归一化坐标      $ 4 = x,y,vix,sigma    x,y取值0~1
        joints[:,:2] = (joints[:,:2] - person_boxs[:,None,:2])/person_boxs[:,None,2:4]     
        heatmaps = self.get_heatmap(joints)
        """
        # tensor:
        #   img.shape = (3,input_size,input_size)
        #   person_boxs.shape = (n_person,4)        # (x1,y1,w,h)
        #   heatmaps.shape = (n_person,17,h,w)      # h,w = output_heatmap_size = 56
        """
        return img , person_boxs , heatmaps


    def get_joints(self,annos):
        num_person = len(annos)
        if self.with_scale_sigma:
            joints = np.zeros((num_person,17,4),dtype=np.float32)
        else:
            joints = np.zeros((num_person,17,3),dtype=np.float32)

        for idx , person in enumerate(annos):
            person_joints = np.array(person["keypoints"]).reshape(-1,3)
            joints[idx,:,:3] = person_joints
            if self.with_scale_sigma:
                box = person["bbox"]
                box_area = box[2]*box[3]
                n_joints = np.sum(person_joints>0)/3                 # 计算该人有几个关键点
                area_adjusted = box_area/(n_joints/17)
                ratio = float(np.sqrt(area_adjusted)/self.image_size)
                sigma = ratio * self.base_sigma
                joints[idx,:,3] = sigma

        return joints

    def adjust_box(self,annos,img_h,img_w):
        """
        将人体框限制在图片范围内
        将 person_box 调整到相同的宽高比  
        """
        num_person = len(annos)
        boxs = np.zeros((num_person,4))
        for i , anno in enumerate(annos):
            x,y,w,h = anno["bbox"]
            x1 , y1 = max(0,x) , max(0,y)
            x2 , y2 = min(img_w,x+w) , min(img_h,y+h)
            c_x , c_y = (x1+x2)/2 , (x2+y2)/2
            w , h = x2-x1 , y2-y1
            if w/h > self.hw_ratio :
                h = w/self.hw_ratio
            else:
                w = h*self.hw_ratio
            x1 , y1 = max(0,c_x - w/2) , max(0,c_y - h/2)
            x2 , y2 = min(img_w,c_x + w/2) , min(img_h,c_y + h/2)
            boxs[i,:] = [x1,y1, x2-x1,y2-y1]

        return boxs

    def transform_fn(self,img,person_boxs,joints):
        h,w,_ = img.shape
        ratio_h = self.image_size/h 
        ratio_w = self.image_size/w 
        img = self.get_transforms(img)      # resize , totensor , # (3,image_size,image_size)
        person_boxs[:,0:4:2] = person_boxs[:,0:4:2]*ratio_w
        person_boxs[:,1:4:2] = person_boxs[:,1:4:2]*ratio_h
        person_boxs = torch.from_numpy(person_boxs)
        joints[...,:2] = joints[...,:2]*[ratio_w,ratio_h]
        joints = torch.from_numpy(joints)

        return img , person_boxs , joints



