# from torchvision.ops import RoIAlign
# import torch
# import numpy as np
# import cv2
# img = cv2.imread(r'C:\Users\wmh_w\Pictures\Camera Roll\34.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = np.transpose(img,axes=(2,0,1))
# img2 = cv2.imread(r'C:\Users\wmh_w\Pictures\Camera Roll\flip34.jpg')
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
# img2 = np.transpose(img2,axes=(2,0,1))
# imgs = np.stack((img,img2),axis=0)
# roi_fn = RoIAlign((14,14),spatial_scale=1.0,sampling_ratio=2)
# imgss = torch.from_numpy(imgs)
# boxes = torch.tensor([[200,200,400,400,0],[100,300,400,400,1]])
# boxes = boxes.float()
# imgss=imgss.float()
# outs = roi_fn(imgss,boxes)
# print(outs[0].shape)
# print(outs[1].shape)


from Models.model import E2EModel
from utils.config import Cfg

model = E2EModel(Cfg)
