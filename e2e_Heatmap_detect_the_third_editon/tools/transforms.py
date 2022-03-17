import cv2
import numpy as np
import torchvision.transforms as T

class GetTransforms():
    def __init__(self,input_size,):
        self.input_size = input_size
        self.transforms_list = T.Compose([
            T.ToTensor(),
        ])
    

    def __call__(self,img):
        img = cv2.resize(img,(self.input_size,self.input_size),interpolation=cv2.INTER_NEAREST)
        img = self.transforms_list(img)

        return img