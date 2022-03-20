import numpy as np

class Generate_heatmap:
    def __init__(self,num_joints,output_size,base_sigma,with_scale_sigma=False):
        self.base_sigma = base_sigma
        self.with_scale_sigma = with_scale_sigma
        self.output_size = output_size
        self.num_joints = num_joints
        if not with_scale_sigma:
            ksize = int(6*base_sigma + 1)
            kx = np.arange(0,ksize,1,dtype=float)
            ky = kx[:,np.newaxis]
            kernal_r = ksize//2
            self.kernal_r = kernal_r 
            x0 = kernal_r
            y0 = kernal_r
            self.gauss_kernal = np.exp(-((kx-x0)**2+(ky-y0)**2)/(2*base_sigma**2))

    def __call__(self,joints):
        """
        # args:
        #   joints.shape = (n_person,17,4) or (n,17,3)   $ 4 = x,y,vis,sigma       # x,y取值0~1
        # return:
        #   heatmap.shape = (n_person,17,output_size,output_size)
        """
        num_person = joints.shape[0]
        joints[...,:2] *= self.output_size
        heatmap = np.zeros((num_person,self.num_joints,self.output_size,self.output_size),dtype=np.float32)
        for i , person_joints in enumerate(joints):
            if self.with_scale_sigma:
                sigma = person_joints[0,-1]
                ksize = int(6*sigma + 1)
                kx = np.arange(0,ksize,1,dtype=float)
                ky = kx[:,np.newaxis]
                kernal_r = ksize//2
                self.kernal_r = kernal_r 
                x0 = kernal_r
                y0 = kernal_r
                self.gauss_kernal = np.exp(-((kx-x0)**2+(ky-y0)**2)/(2*sigma**2))
            for j , joint in enumerate(person_joints):
                if joint[2] <= 0:
                    continue
                x , y = int(joint[0]) , int(joint[1])
                if x < 0 or y < 0 or \
                        x >= self.output_size or y >= self.output_size:
                        continue
                tl , br = (x-self.kernal_r , y-self.kernal_r) , (x+self.kernal_r,y+self.kernal_r)

                # 计算高斯核与图像的交集部分，该部分在高斯核的那一区域 
                k_x1 , k_y1 = max(0,tl[0])-tl[0] , max(0,tl[1])-tl[1]
                k_x2 , k_y2 = min(self.output_size,br[0])-tl[0] , min(self.output_size,br[1])-tl[1]
                # 高斯核与图像交集部分，该部分在热图像的那一区域
                h_x1 , h_y1 = max(0,tl[0]) , max(0,tl[1])
                h_x2 , h_y2 = min(self.output_size,br[0]) , min(self.output_size,br[1])

                heatmap[i,j,h_y1:h_y2,h_x1:h_x2] = self.gauss_kernal[k_y1:k_y2,k_x1:k_x2]
            
        return heatmap