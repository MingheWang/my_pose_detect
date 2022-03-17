import cv2
import os
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoDataset(Dataset):
    def __init__(self, root, anno_name):
        self.root = root
        self.anno_name = anno_name
        self.dataset_name = anno_name.split(".")[0].split("_")[-1]
        self.anno_file_path = os.path.join(root , "annotations" , "{}.json".format(anno_name))
        self.coco = COCO(self.anno_file_path)
        # self.ids = list(self.coco.imgs.keys())
        self.img_ids = self.coco.getImgIds()
        self.totensor = ToTensor()

    def __len__(self,):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)            # list
        img_name = self.coco.imgs[img_id]["file_name"]
        img_path = os.path.join(self.root,self.dataset_name,img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        return img , annos

    def rand(self,x=0,y=1):
        return x + np.random.rand()*(y-x)

    