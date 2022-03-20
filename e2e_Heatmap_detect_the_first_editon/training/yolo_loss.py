from multiprocessing import reduction
from nis import match
import torch
import torch.nn as nn
import numpy as np
import math
from utils.decode_yolo import YoloDecoder
from utils.get_iou import Get_box_iou
import torch.nn.functional as F

class YoloLoss:
    def __init__(self,cfg):
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_size
        self.num_classes = cfg.num_classes  # 80
        self.anchors = cfg.anchors          # [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        self.anchors = torch.tensor(self.anchors).reshape(-1,2)     # (9,2)
        self.strides = cfg.strides          # [8,16,32]
        self.output_sizes = [cfg.image_size // self.strides[i] for i in range(len(cfg.strides))]      # [416/8,416/16,416/32]  
        self.yolo_ignore_conf = cfg.yolo_ignore_thre    # 0.3
        self.yolo_with_obj_thre = cfg.yolo_thre_with_obj    # 0.5
        self.yolo_decoder = YoloDecoder(cfg,training=True,training_yolo=True)

    def get_target(self,bboxes,targets,obj_mask,target_mask):
        """
        args:
            bboxes:   [(n1,5),(n2,5),....,(nb,5)]      # (x1,y1,w,h,id) 
            targets:    [zeros(b,3,h,w,5+class),zeros(b,3,h,w,5+class),zeros(b,3,h,w,5+class)] 
        """ 
        gt_boxes = [bbox.clone() for bbox in bboxes]

        anchors = torch.cat((torch.zeros_like(self.anchors),self.anchors),dim=1)        # (9,4)   0,0,w,h
        for i , boxes in enumerate(gt_boxes):
            if torch.sum(boxes)==0:
                continue
            # boxes: shape=(ni,5)
            boxes[:,:2] = boxes[:,:2] + boxes[:,2:4]/2      # 5 = cx,xy,w,h
            boxes_ = torch.zeros((boxes.shape[0],4),dtype=torch.float32)
            boxes_[:,2:4] = boxes[:,2:4]    # 0,0,w,h
            iou_boxes2anchor = Get_box_iou(boxes_,anchors)      # (ni,9)
            # ind.shape = (ni,)
            _ , ind = torch.max(iou_boxes2anchor,dim=1)
            fmap_ids = torch.div(ind,3,rounding_mode='floor').int()
            anchor_ids = ind%3
            for b_i , box in enumerate(boxes):
                fmap_id = fmap_ids[b_i]         
                anchor_id = anchor_ids[b_i]                     # gai mu biao kuang dui ying de di fmap_id ge te zheng tu di anchor_id ge mao kuang.

                box[:4] = box[:4] / self.strides[fmap_id]         # cx,xy,w,h
                class_id = box[4].int()
                xi , yj = box[0].int().item() , box[1].int().item()

                targets[fmap_id][i,anchor_id,yj,xi,:4] = box[:4] * self.strides[fmap_id]
                targets[fmap_id][i,anchor_id,yj,xi,4] = 1.0
                targets[fmap_id][i,anchor_id,yj,xi,5+class_id] = 1.0

                target_mask[fmap_id][i,anchor_id,yj,xi] = 1.0
                obj_mask[fmap_id][i,anchor_id,yj,xi] = 1.0

            return targets , obj_mask , target_mask

    def get_no_obj(self,gt_boxes,yolo2outs,noobj_mask,obj_mask):
        """
        args:
            gt_boxes:   [(n1,5),(n2,5),....,(nb,5)]      # (x1,y1,w,h)       # 原尺寸
            yolo2outs:  [(b,3,h,w,5+class),(b,3,h,w,5+class),(b,3,h,w,5+class)]     # (cx,cy,w,h)解码后在对应输入图片上的坐标
            noobj_mask: [(b,3,h,w),(b,3,h,w),(b,3,h,w)]
            obj_mask: [(b,3,h,w),(b,3,h,w),(b,3,h,w)]
        """
        yolo_out_list = [out.clone() for out in yolo2outs]

        fmap1_size = noobj_mask[0].shape[1:]        # 3,h,w
        fmap1_long = torch.prod(torch.tensor(fmap1_size))         # 3*h*w
        fmap2_size = noobj_mask[1].shape[1:]
        fmap2_long = torch.prod(torch.tensor(fmap2_size))
        fmap3_size = noobj_mask[2].shape[1:]
        fmap3_long = torch.prod(torch.tensor(fmap3_size))

        for i in range(len(yolo_out_list)):
            # reshape output:   (b,3,h,w,5+cls) ----> (b,3*h*w,5+cls)
            yolo_out_list[i] = yolo_out_list[i].view(self.batch_size,-1,5+self.num_classes).contiguous()    # (b,3*h*w,5+class)
            # cx,xy,w,h ---> x1,y1,w,h
            yolo_out_list[i][...,:2]  = yolo_out_list[i][...,:2] - yolo_out_list[i][...,2:4]/2  # x1,y1,w,h

        # concat three feature maps
        yolo_outs = torch.cat(yolo_out_list,dim=1)      # shape=(b,num1+num2+num3,5+class)
        for b_i in range(len(gt_boxes)):
            bboxes = gt_boxes[b_i][:,:4].clone()             # (ni,4)    # (x1,y1,w,h)       # 原尺寸

            if torch.sum(bboxes)==0:
                continue

            # (x1,y1,w,h) ---> (x1,y1,x2,y2)
            bboxes[:,2:4] = bboxes[:,:2] + bboxes[:,2:4]     # (x1,y1,x2,y2)       # 原尺寸

            # (x1,y1,w,h) ---> (x1,y1,x2,y2)
            yolo_out = yolo_outs[b_i,:,:4]         # (num1+num2+num3,4)   # (x1,y1,w,h)       # 原尺度
            yolo_out[:,2:4] = yolo_out[:,:2] + yolo_out[:,2:4]     # (x1,y1,x2,y2)       # 原尺寸
            yolo_out[:,:4] = torch.clamp(yolo_out[:,:4],min=0,max=self.image_size)

            iou_yolo_out2bboxes = Get_box_iou(yolo_out,bboxes)     # (num,n)
            iou_val , _ = torch.max(iou_yolo_out2bboxes,dim=1)     # (num,) , (num,)

            none_mask = iou_val < self.yolo_ignore_conf      # (num,)
            none_mask1 = none_mask[:fmap1_long].reshape(fmap1_size)           # 3,h,w
            none_mask2 = none_mask[fmap1_long:fmap1_long+fmap2_long].reshape(fmap2_size)      # 3,h,w
            none_mask3 = none_mask[fmap1_long+fmap2_long:].reshape(fmap3_size) # 3,h,w
            noobj_mask[0][b_i] = none_mask1
            noobj_mask[1][b_i] = none_mask2
            noobj_mask[2][b_i] = none_mask3

            mask = iou_val >= self.yolo_with_obj_thre
            mask1 = mask[:fmap1_long].reshape(fmap1_size)           # 3,h,w
            mask2 = mask[fmap1_long:fmap1_long+fmap2_long].reshape(fmap2_size)      # 3,h,w
            mask3 = mask[fmap1_long+fmap2_long:].reshape(fmap3_size) # 3,h,w
            obj_mask[0][b_i] = mask1
            obj_mask[1][b_i] = mask2
            obj_mask[2][b_i] = mask3

        return noobj_mask , obj_mask

    def iou_loss(self,predict_boxes,target_boxes):
        """
            predict_boxes:  [n,4]       # x1,y1,x2,y2
            target_boxes:   [n,4]       # x1,y1,x2,y2
        """
        predict_boxes_wh = predict_boxes[:,2:4] - predict_boxes[:,:2]   # (n,2)
        target_boxes_wh = target_boxes[:,2:4] - target_boxes[:,:2]

        predict_boxes_center = (predict_boxes[:,:2] + predict_boxes[:,2:4])/2
        target_boxes_center = (target_boxes[:,:2] + target_boxes[:,2:4])/2
        center_distance = torch.sum(torch.pow((predict_boxes_center - target_boxes_center), 2), dim=1)

        inter_lts  = torch.max(predict_boxes[:,:2], target_boxes[:,:2])
        inter_rbs = torch.min(predict_boxes[:,2:4], target_boxes[:,2:4])
        inter_wh = torch.max(inter_rbs - inter_lts, torch.zeros_like(inter_lts))
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        pred_area = torch.prod(predict_boxes_wh ,dim=1)
        target_area = torch.prod(target_boxes_wh ,dim=1)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / torch.clamp(union_area,min = 1e-6)

        enclose_lts = torch.min(predict_boxes[:,:2], target_boxes[:,:2])
        enclose_rbs = torch.max(predict_boxes[:,2:4], target_boxes[:,2:4])
        enclose_wh = torch.max(enclose_rbs - enclose_lts, torch.zeros_like(enclose_rbs))

        enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), dim=1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
        
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(predict_boxes_wh[:, 0] / torch.clamp(predict_boxes_wh[:, 1],min = 1e-6)) - torch.atan(target_boxes_wh[:, 0] / torch.clamp(target_boxes_wh[:, 1], min = 1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou



    def __call__(self,yolo_out_lists,bboxes):
        """
        after yolo_decoder:
            yolo_out_lists:     [(b,3*(5+class),h,w),(b,3*(5+class),h,w),(b,3*(5+class),h,w)]       # yolo output
            yolo_out_list:      [(b,3,h,w,5+class),(b,3,h,w,5+class),(b,3,h,w,5+class)]     # (cx,cy,w,h)解码后在对输入图片上的坐标
        bboxes:    [(n1,5),(n2,5),....,(nb,5)]      # (x1,y1,w,h)       # 原尺寸
        """
        yolo_out_list = self.yolo_decoder(*yolo_out_lists)
        targets = [torch.zeros_like(yolo_out_list[i]) for i in range(len(yolo_out_list))]      # (b,3,h,w,5+class)
        noobj_mask = [torch.zeros(targets[i].shape[:-1],dtype=bool) for i in range(len(targets))]         # (b,3,h,w)
        obj_mask = [torch.zeros(targets[i].shape[:-1],dtype=bool) for i in range(len(targets))]           # (b,3,h,w)
        target_mask = [torch.zeros(targets[i].shape[:-1],dtype=bool) for i in range(len(targets))]           # (b,3,h,w)
        noobj_mask , obj_mask = self.get_no_obj(bboxes,yolo_out_list,noobj_mask,obj_mask)                   
        targets , obj_mask , target_mask = self.get_target(bboxes,targets,obj_mask,target_mask)
        
        select_yolo_out = torch.cat([yolo_out_list[i][target_mask[i]] for i in range(len(yolo_out_list))],dim=0)          # (n1+n2+n3,5+class)
        select_target = torch.cat([targets[i][target_mask[i]] for i in range(len(yolo_out_list))],dim=0)                  # (n1+n2+n3,5+class)
        
        predict_boxes = select_yolo_out[:,:4]       # (n1+n2+n3,4)      on the scale of original image
        target_boxes = select_target[:,:4]          # (n1+n2+n3,4)      on the scale of original image
        
        perdict_cls_conf = select_yolo_out[:,5:]    # (n1+n2+n3,n_class)
        target_cls_conf = select_target[:,5:]

        # (cx,cy,w,h) ---> (x1,y1,x2,y2)
        predict_boxes[:,:2] , predict_boxes[:,2:4] = predict_boxes[:,:2] - predict_boxes[:,2:4]/2 , predict_boxes[:,:2] + predict_boxes[:,2:4]/2        # x1,y1,x2,y2
        predict_boxes = torch.clamp(predict_boxes,min=0,max=self.image_size)
        target_boxes[:,:2] , target_boxes[:,2:4] = target_boxes[:,:2] - target_boxes[:,2:4]/2 , target_boxes[:,:2] + target_boxes[:,2:4]/2        # x1,y1,x2,y2

        predict_obj_conf = torch.cat([yolo_out_list[i][obj_mask[i]][:,4] for i in range(len(yolo_out_list))],dim=0)     # (m1+m2+m3,)
        predict_noobj_conf = torch.cat([yolo_out_list[i][noobj_mask[i]][:,4] for i in range(len(yolo_out_list))],dim=0)     # (k1+k2+k3,)

        # cls_loss = - target_cls_conf * torch.log(perdict_cls_conf) - (1.0 - target_cls_conf) * torch.log(1.0 - perdict_cls_conf)
        # obj_loss = - torch.log(predict_obj_conf)
        # noobj_loss = - torch.log(1-predict_noobj_conf)
        obj_loss = F.binary_cross_entropy(predict_obj_conf,torch.ones_like(predict_obj_conf),reduction="sum")
        noobj_loss = F.binary_cross_entropy(predict_noobj_conf,torch.zeros_like(predict_noobj_conf),reduction="sum")
        cls_loss = F.binary_cross_entropy(perdict_cls_conf,target_cls_conf,reduction="sum")
        iou_loss = torch.sum(1 - self.iou_loss(predict_boxes,target_boxes))

        # cls_loss_all = torch.sum(cls_loss)
        # obj_loss_all = torch.sum(obj_loss)
        # noobj_loss_all = torch.sum(noobj_loss)
        # iou_loss_all = torch.sum(iou_loss)


        print("\n")
        print(f"cls_loss_all    ={cls_loss}")
        print(f"obj_loss_all    ={obj_loss}")

        print(f"noobj_loss_all  ={noobj_loss}")
        print(f"iou_loss_all    ={iou_loss}")

        loss_all = cls_loss + obj_loss + iou_loss + noobj_loss

        return loss_all


class originloss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super(originloss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask
        self.label_smoothing = label_smoothing

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
        
    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   求真实框和预测框所有的iou
        #----------------------------------------------------#
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / torch.clamp(union_area,min = 1e-6)

        #----------------------------------------------------#
        #   计算中心的差距
        #----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
        
        #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        #   计算对角线距离
        #----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
        ciou            = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
        
        v       = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min = 1e-6))), 2)
        alpha   = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou    = ciou - alpha * v
        return ciou

           #---------------------------------------------------#
    #   平滑标签
    #---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        #----------------------------------------------------#
        #   l 代表使用的是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets 真实框的标签情况 [batch_size, num_gt, 5]
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #--------------------------------#
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3 * (5+num_classes), 13, 13 => bs, 3, 5 + num_classes, 13, 13 => batch_size, 3, 13, 13, 5 + num_classes

        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-----------------------------------------------#
        #   获得网络应该有的预测结果
        #-----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.cuda()
            noobj_mask      = noobj_mask.cuda()
            box_loss_scale  = box_loss_scale.cuda()
        #-----------------------------------------------------------#
        #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
        #   表示真实框的宽高，二者均在0-1之间
        #   真实框越大，比重越小，小框的比重更大。
        #-----------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        #---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        #----------------------------------------------------------------#
        ciou        = (1 - self.box_ciou(pred_boxes[y_true[..., 4] == 1], y_true[..., :4][y_true[..., 4] == 1])) * box_loss_scale[y_true[..., 4] == 1]
        loss_loc    = torch.sum(ciou)
        #-----------------------------------------------------------#
        #   计算置信度的loss
        #-----------------------------------------------------------#
        loss_conf   = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                      torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)

        loss_cls    = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))

        loss        = loss_loc + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs              = len(targets)
        #-----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        #-----------------------------------------------------#
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   让网络更加去关注小目标
        #-----------------------------------------------------#
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #-----------------------------------------------------#
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):            
            if len(targets[b])==0:
                continue
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            #-------------------------------------------------------#
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            #-------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                #----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框
                #----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                #----------------------------------------#
                #   获得真实框属于哪个网格点
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   取出真实框的种类
                #----------------------------------------#
                c = batch_target[t, 4].long()
                
                #----------------------------------------#
                #   noobj_mask代表无目标的特征点
                #----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                #----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #----------------------------------------#
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale
    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        for b in range(bs):           
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4]
                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)