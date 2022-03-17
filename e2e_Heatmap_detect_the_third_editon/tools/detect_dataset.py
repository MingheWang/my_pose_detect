from random import sample
import cv2
import os
from tools.create_cocodataset import CocoDataset
import torch

import numpy as np

class Detect_Dataset(CocoDataset):
    def __init__(self,root, anno_name, image_size, use_mosaic=False, is_train=False):
        super().__init__(root, anno_name)
        self.image_size = image_size
        self.use_mosaic = use_mosaic
        self.is_train = is_train

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        
        if self.is_train:
            if self.use_mosaic and self.rand() > 0.5 :
                img_id_list = sample(self.img_ids,3)
                img_id_list.append(img_id)
                # image: shape=(image_size,image_size,3)
                # box:  shape=(n,5)     # 5 = x1,y1,w,h,id
                img , box = self.get_data_with_Mosaic(img_id_list,self.image_size)

                img = self.totensor(img)
                box = torch.from_numpy(box)
                # img: tensor(3,image_size,image_size)
                # box: tensor(n,5)      # 5 = x1,y1,w,h,id
                return img , box
            else:
                anno_ids = self.coco.getAnnIds(imgIds=img_id)
                annos = self.coco.loadAnns(anno_ids)            # list
                # box = np.array([map(int,anno["bbox"]) for anno in annos if anno["iscrowd"]==0])      # x1,y1,w,h
                # cat_id = np.array(map(int,[anno["category_id"] for anno in annos if anno["iscrowd"]==0])).reshape(-1,1)
                box = []
                cat_id = []
                for anno in annos:
                    box.append(anno["bbox"])
                    cat_id.append(self.adjust_catid(anno["category_id"]))
                if len(box)==0:
                    box = np.zeros((1,5))
                else:
                    box = np.array(box)
                    maskw , maskh = box[:,2]>1 , box[:,3]>1
                    mask = maskw*maskh
                    cat_id = np.array(cat_id).reshape(-1,1)
                    box = np.concatenate((box,cat_id),axis=1)        # x1,y1,w,h,id
                    box = box[mask]

                img_name = self.coco.imgs[img_id]["file_name"]
                img_path = os.path.join(self.root,self.dataset_name,img_name)
                img = cv2.imread(img_path)
                ih,iw,_ = img.shape
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_NEAREST)
                box[:,2:4] = box[:,:2] + box[:,2:4]             # x1,y1,w,h,id ---> x1,y1,x2,y2,id
                box[:, [0,2]] = box[:, [0,2]]*(self.image_size/iw)
                box[:, [1,3]] = box[:, [1,3]]*(self.image_size/ih)
                box = torch.clamp(torch.tensor(box),min=0,max=self.image_size)
                box[:,2:4] = box[:,2:4] - box[:,:2]             # x1,y1,x2,y2,id ---> x1,y1,w,h,id

                """
                img: shape=(3,image_size,image_size)
                box: shape=(n,5)        # 5 = x1,y1,w,h,id
                """
                img = self.totensor(img)
                
                return img , box
            
        else:
            anno_ids = self.coco.getAnnIds(imgIds=img_id)
            annos = self.coco.loadAnns(anno_ids)            # list
            # box = np.array([map(int,anno["bbox"]) for anno in annos if anno["iscrowd"]==0])      # x1,y1,w,h
            # cat_id = np.array(map(int,[anno["category_id"] for anno in annos if anno["iscrowd"]==0])).reshape(-1,1)
            # box = np.concatenate((box,cat_id),axis=1)        # x1,y1,w,h,id
            box = []
            cat_id = []
            for anno in annos:
                if anno["iscrowd"] == 1:
                    continue
                box.append(anno["bbox"])
                cat_id.append(
                    self.adjust_catid(anno["category_id"]))
            box = np.array(box)
            maskw , maskh = box[:,2]>1 , box[:,3]>1
            mask = maskw*maskh
            cat_id = np.array(cat_id).reshape(-1,1)
            box = np.concatenate((box,cat_id),axis=1)        # x1,y1,w,h,id
            box = box[mask]

            img_name = self.coco.imgs[img_id]["file_name"]
            img_path = os.path.join(self.root,self.dataset_name,img_name)
            img = cv2.imread(img_path)
            ih,iw,_ = img.shape

            scale = min(self.image_size/iw, self.image_size/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (self.image_size-nw)//2
            dy = (self.image_size-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            img = cv2.resize(img,(nw,nh), interpolation=cv2.INTER_NEAREST)
            new_img = np.ones((self.image_size,self.image_size,3))*128
            new_img[dy:dy+nh,dx:dx+nw] = img

            if len(box)>0:
                np.random.shuffle(box)
                box[:,2:4] = box[:,:2] + box[:,2:4]             # x1,y1,w,h,id ---> x1,y1,x2,y2,id  
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>self.image_size] = self.image_size
                box[:, 3][box[:, 3]>self.image_size] = self.image_size
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
                box[:,2:4] = box[:,2:4] - box[:,:2]             # x1,y1,x2,y2,id ---> x1,y1,w,h,id

            """
            img: shape=(3,image_size,image_size)
            box: shape=(n,5)        # 5 = x1,y1,w,h,id
            """
            img = self.totensor(img)
            box = torch.from_numpy(box)
            return new_img, box



    def get_data_with_Mosaic(self, img_id_list, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):


        h, w = input_shape , input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)
        # 四张图片放缩后的 w 和 h   
        nws     = [ int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]
        nhs     = [ int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]
        
        place_x = [int(w*min_offset_x) - nws[0], int(w*min_offset_x) - nws[1], int(w*min_offset_x), int(w*min_offset_x)]
        place_y = [int(h*min_offset_y) - nhs[0], int(h*min_offset_y), int(h*min_offset_y), int(h*min_offset_y) - nhs[3]]
        
        image_lists = [] 
        box_lists   = []
        index = 0
        for img_id in img_id_list:
            anno_ids = self.coco.getAnnIds(imgIds=img_id)
            annos = self.coco.loadAnns(anno_ids)            # list
            img_name = self.coco.imgs[img_id]["file_name"]
            img_path = os.path.join(self.root,self.dataset_name,img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            # 图片的大小
            ih,iw,_ = img.shape
            box = []
            cat_id = []
            for anno in annos:
                if anno["iscrowd"] == 1:
                    continue
                box.append(anno["bbox"])
                cat_id.append(self.adjust_catid(anno["category_id"]))
            box = np.array(box)
            maskw , maskh = box[:,2]>1 , box[:,3]>1
            mask = maskw*maskh
            cat_id = np.array(cat_id).reshape(-1,1)
            box = np.concatenate((box,cat_id),axis=1)        # x1,y1,w,h,id
            box = box[mask]

            # 是否翻转图片
            if self.rand()<.5 and len(box)>0:
                img = img[:,::-1]
                box[:, 0] = iw - box[:,0] - box[:,2]

            nw = nws[index]             #   
            nh = nhs[index]             #   
            img = cv2.resize(img,(nw,nh), interpolation=cv2.INTER_NEAREST)

            # ----------------------------------------------------------------- #
            # 将图片进行放置，分别对应四张分割图片的位置
            # ----------------------------------------------------------------- #
            dx = place_x[index]         
            dy = place_y[index]             # (dx,dy)为缩放后将该图片的左上角放置在(w,h)中的位置
            new_image = np.ones((h,w,3))*128
            new_image[dy:dy+nh,dx:dx+nw] = img

            index = index + 1
            # box_data = []
            # ------------------------------------------------------------------ #
            # 对box进行重新处理，和图片相同的处理（放缩），根据放置位置再平移目标框。           
            # ------------------------------------------------------------------ #  
            if len(box)>0:
                np.random.shuffle(box)
                box[:,2:4] = box[:,:2] + box[:,2:4]             # x1,y1,w,h,id ---> x1,y1,x2,y2,id
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_lists.append(new_image)
            box_lists.append(box_data)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_lists[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_lists[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_lists[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_lists[3][:cuty, cutx:, :]

        # 进行色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(new_image/255,np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_lists, cutx, cuty)

        """
        new_image: shape=(h,w,3)
        new_boxes: shape=(n,5)      # 5 = x1,y1,w,h,id
        """
        return new_image, new_boxes


    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2-x1)
                tmp_box.append(y2-y1)
                tmp_box.append(box[-1])             # tmp_box: list([x1,y1,w,h,id])
                merge_bbox.append(tmp_box)          # merg_bbox: boxlist(list)
        merge_bbox = np.array(merge_bbox)
        return merge_bbox



    def adjust_catid(self,cat):

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        
        return cat