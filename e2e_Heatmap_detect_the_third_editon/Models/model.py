from re import I
import numpy as np
import torch
from torchvision.ops import RoIAlign
from torch import nn
from collections import OrderedDict
from Models.CSPdarknet import CSPDarkNet 
from Models.pose_layers import batchnorm
from Models.detect_net import DetectNet
from Models.posenet import PoseNet
from utils.decode_yolo import YoloDecoder
from tools.get_features import adjust_predict_person_boxes_with_a_gaven_scale


############################################################################################################################
############################################################################################################################
class E2EModel(nn.Module):
    def __init__(self,cfg):
        super(E2EModel,self).__init__()
        self.training = cfg.training
        if self.training:
            self.training_yolo = cfg.training_yolo
            self.training_pose = cfg.training_pose
            assert self.training_yolo != self.training_pose , "only train a part net at a moment"
        else:
            self.training_yolo = False
            self.training_pose = False

        self.backbone = CSPDarkNet(cfg.yolo_bk_layers)
        self.detect_net = DetectNet(cfg)
        if self.training:
            if self.training_yolo:
                self.decode_yolo = YoloDecoder(cfg,self.training,training_yolo=self.training_yolo)
        else:       # 对训练好的模型进行推理
            self.decode_yolo = YoloDecoder(cfg)

        self.hw_ratio = cfg.hw_ratio
        # the box coordinates in (x1, y1, x2, y2) format
        # boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2) format 
        # where the regions will be taken from.
        # If a single Tensor is passed, then the first column should
        # contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
        # If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i in the batch.
        if not self.training or self.training_pose:
            self.roi_heads = [RoIAlign(output_size=(14,14),spatial_scale=8,sampling_ratio=2),RoIAlign(output_size=(14,14),spatial_scale=16,sampling_ratio=2)] 
            self.posenet = PoseNet(nstack=cfg.nstack,in_dim=cfg.posnet_input_dim,oup_dim=cfg.num_joints)

    def forward(self,x,gt_person_boxes=None ,gt_heatmaps=None):
        """
        ###############################################################################
        #   x:  (b_size,3,h,w)
        #   gt_person_boxes:   [(n_person,4),(n_person,4),...]  , len(list) = batch_size
        #   gt_heatmaps:       [(n_person,17,H=56,W=56),(n_person,17,H,W),...]
        ###############################################################################
        """
        #------------------------------------------#
        #  backbone
        # x2:   B,256,52,52
        # x1:   B,512,26,26
        # x0:   B,1024,13,13
        #------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        out2,out1,out0 = self.detect_net(x2,x1,x0)
        
        if self.training:
            if self.training_yolo:
                #----------------------------------------------------------------------#
                # if self.training_yolo:    得到预测框在输入图片上的真实位置xy和wh
                #   out1:   tensor.shape=(b_size , c_size , h_size1 , w_size1 , n_attr)
                #   out2:   tensor.shape=(b_size , c_size , h_size2 , w_size2 , n_attr)
                #   out3:   tensor.shape=(b_size , c_size , h_size3 , w_size3 , n_attr)
                #----------------------------------------------------------------------#
                out1,out2,out3 = self.decode_yolo(out2,out1,out0)
                return [out1,out2,out3]
            
            elif self.training_pose:
                #------------------------------------------------------------------------------------------------------#
                # gt_person_boxes:
                #    [tensor(num_person1,4) , (num_person2,4) , ....]  ,  len = b_size;      $$ 4 = x1 , y1 , w , h
                # 如果该图片中没有人体框，则直接令其为 torch.zeros(1,4)
                # gt_heatmaps:
                #    [tensor(num_person1,17,h,w),...]
                #------------------------------------------------------------------------------------------------------#
                gt_heatmaps = torch.cat(gt_heatmaps,dim=0)      # (K,17,56,56)      K = num1+num2+num3+...

                # [num_person1 , num_person2 , ... , numperson_b] ， 用于保存每张图片中的人数
                num_boxes_per_img = [ boxes.shape[0] for boxes in gt_person_boxes]
                # [True,True,False,...]，用于判断每张图片是否有人
                img_with_person = [ torch.sum(boxes)>0 for boxes in gt_person_boxes]

                mask = np.array(img_with_person).repeat(num_boxes_per_img,axis=0)

                img_id = np.arange(len(gt_person_boxes))
                id_repeats = np.repeat(img_id, num_boxes_per_img)        # (K,)
                id_repeats = torch.from_numpy(id_repeats)
                person_boxes = torch.cat(gt_person_boxes,dim=0)     # (K,4)     4 = x1,y1,w,h
                person_boxes[:,2:4] = person_boxes[:,:2] + person_boxes[:,2:4]      # 4 = x1,y1,x2,y2
                box_with_img_id = torch.cat((id_repeats,person_boxes),dim=1)    # (K,5)     # 5 = img_id,x1,y1,x2,y2
                box_with_img_id = box_with_img_id[mask]         # (K1,5)     # 5 = img_id,x1,y1,x2,y2

                gt_heatmaps = gt_heatmaps[mask]     # (K1,17,56,56)
                img_ids = box_with_img_id[:,0]      # (K1,)

                # x2_out为：shape=(K1,256,14,14)
                x2_out = self.roi_heads[0](x2 , box_with_img_id)       
                # x1_out特征图为：shape=(K1,512,14,14)
                x1_out = self.roi_heads[1](x1 , box_with_img_id)       

                out_per_boxes = []          # [(768,14,14),(768,14,14),...]
                for k in range(len(x2_out.shape[0])):       # 第 k 个人
                    temp = torch.cat((x2_out[k],x1_out[k]),dim=0)       # shape=(256+512,14,14)
                    out_per_boxes.append(temp)

                # input_features: shape=(K1,768,14,14)       # K1 = 一个batch_size的人数
                input_features = torch.stack(out_per_boxes,dim=0)

                # (K,768,14,14) ---> pose_out: shape=(K1 , num_joints=17 , fsize=56 , fsize=56)
                pose_out = self.posenet(input_features)            

                """
                return:
                    pose_out:   shape=(K1 , num_joints=17 , fsize=56 , fsize=56)
                    gt_heatmaps:    shape=(K1,17,56,56)    
                """
                return pose_out  , gt_heatmaps

        # not self.training:    
        else:
            #-----------------------------------------------------------------------------------#
            # person_boxes_with_nms: [(n1 , 4),(n2 , 4) , ...]          # 4 = cx , cy , w , h
            # 如果该图片中没有人体框，则直接令其为 torch.zeros(1,4)
            #-----------------------------------------------------------------------------------#
            perosn_boxes_with_nms = self.decode_yolo(out2.detach(),out1.detach(),out0.detach())
            #------------------------------------------------------------------------------------------------------#
            # predict_person_boxes:
            #    [tensor(num_person1,4) , (num_person2,4) , ....]  ,  len = b_size;      $$ 4 = x1 , y1 , x2 , y2
            # 如果该图片中没有人体框，则直接令其为 torch.zeros(1,4)
            #------------------------------------------------------------------------------------------------------#
            predict_person_boxes = adjust_predict_person_boxes_with_a_gaven_scale(perosn_boxes_with_nms,self.hw_ratio)

            assert self.roi_heads[0].sampling_ratio == x2.shape[-1]//x.shape[-1] , "sampling_ratio Wrong!"
            assert self.roi_heads[1].sampling_ratio == x1.shape[-1]//x.shape[-1] , "sampling_ratio Wrong!"
            
            # [num_person1 , num_person2 , ... , numperson_b] ， 用于保存每张图片中的人数
            num_boxes_per_img = [ boxes.shape[0] for boxes in predict_person_boxes]
            # [True,True,False,...]，用于判断每张图片是否有人
            img_with_person = [ torch.sum(boxes)>0 for boxes in predict_person_boxes]

            mask = np.array(img_with_person).repeat(num_boxes_per_img,axis=0)

            img_id = np.arange(len(predict_person_boxes))
            id_repeats = np.repeat(img_id, num_boxes_per_img)        # (K,)
            id_repeats = torch.from_numpy(id_repeats)
            person_boxes = torch.cat(predict_person_boxes,dim=0)     # (K,4)     4 = x1,y1,w,h
            person_boxes[:,2:4] = person_boxes[:,:2] + person_boxes[:,2:4]      # 4 = x1,y1,x2,y2
            box_with_img_id = torch.cat((id_repeats,person_boxes),dim=1)    # (K,5)     # 5 = img_id,x1,y1,x2,y2
            box_with_img_id = box_with_img_id[mask]         # (K1,5)     # 5 = img_id,x1,y1,x2,y2

            img_ids = box_with_img_id[:,0]      # (K1,)

            # x2_out为：shape=(K1,256,14,14)
            x2_out = self.roi_heads[0](x2 , box_with_img_id)       
            # x1_out特征图为：shape=(K1,512,14,14)
            x1_out = self.roi_heads[1](x1 , box_with_img_id)       

            out_per_boxes = []
            for k in range(len(x2_out.shape[0])):
                temp = torch.cat((x2_out[k],x1_out[k]),dim=0)       # shape=(256+512,14,14)
                out_per_boxes.append(temp)

            # input_features: shape=(K1,768,14,14)       # K1 = 一个batch_size的人数
            input_features = torch.stack(out_per_boxes,dim=0)

            # pose_out: shape=(K1 , num_joints , fsize=56 , fsize=56)
            pose_out = self.posenet(input_features)            

            """
            return:
                pose_out:   shape=(K1 , num_joints , fsize , fsize)
                box_with_img_id:    shape=(K1,5)    # 5 = img_id,x1,y1,x2,y2
            """
            return pose_out, box_with_img_id


    def init_weights(self):
        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name == "Conv2d":
                nn.init.kaiming_normal_(m.weight.data,a=0, mode='fan_in')
            if class_name == "BatchNorm2d":
                nn.init.normal_(m.weight.data,1.0,0.02)
                nn.init.constant_(m.bias.data,0)
        
def get_model(cfg):

    model = E2EModel(cfg)

    if cfg.training:
        model.init_weights()

    return model
