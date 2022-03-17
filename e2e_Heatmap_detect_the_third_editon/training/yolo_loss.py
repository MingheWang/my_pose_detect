from multiprocessing import reduction
from nis import match
import torch
import math
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



    def __call__(self,yolo_out_list,bboxes):
        """
        args:
            yolo_out_list:      [(b,3,h,w,5+class),(b,3,h,w,5+class),(b,3,h,w,5+class)]     # (cx,cy,w,h)解码后在对输入图片上的坐标
            bboxes:    [(n1,5),(n2,5),....,(nb,5)]      # (x1,y1,w,h)       # 原尺寸
        """

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
        iou_loss = 1 - self.iou_loss(predict_boxes,target_boxes)

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