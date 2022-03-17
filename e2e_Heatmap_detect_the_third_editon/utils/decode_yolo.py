import torch
from torchvision.ops import nms
# from nms import nms_fn
from utils.adjust_box import adjust_wh_ratio

class YoloDecoder:
    def __init__(self,cfg,training=False,training_yolo=False):

        self.cfg = cfg
        self.CUDA = cfg.CUDA
        self.training = training
        self.training_yolo = training_yolo
        self.yolo_thre_conf = cfg.yolo_thre_conf
        self.batch_size = cfg.batch_size
        self.bbox_attrs = cfg.bbox_attrs        # 85
        self.point_bbox = cfg.point_bbox        # 3
        self.NMS = nms
        self.adjust_box_wh = adjust_wh_ratio
        self.wh_ratio = cfg.wh_ratio
        self.image_size = cfg.image_size
        self.num_classes = cfg.num_classes  # 80
        self.anchors = cfg.anchors          # [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        self.strides = cfg.strides          # [8,16,32]
        self.output_sizes = [cfg.image_size // self.strides[i] for i in range(len(cfg.strides))]      # [416/8,416/16,416/32]  
        # [(b,3,h1,w,4),(b,3,h2,w,4),(b,3,h3,w,4)]
        self.base_anchor_maps = self.get_anchor_base(self.anchors,self.output_sizes,self.strides,cfg.batch_size)

    def get_anchor_base(self,anchors,output_sizes,strides,batch_size):
        anchors = torch.tensor(anchors,dtype=torch.float32).reshape(-1,2)         # shape=(9,2)
        anchor_maps = []
        for i in range(len(output_sizes)):
            output_size = output_sizes[i]
            anchor = anchors[3*i : 3*(i+1)] / strides[i]       # shape=(3,2)    调整到对应特征图尺度
            grid_x = torch.arange(output_size,dtype=torch.float32).repeat(batch_size,3,output_size,1).unsqueeze(dim=4)
            grid_y = torch.arange(output_size,dtype=torch.float32).repeat(batch_size,3,output_size,1).transpose(dim0=2,dim1=3).unsqueeze(dim=4)
            grid_wh = anchor.repeat(batch_size,output_size,output_size,1,1).permute(0,3,1,2,4)
            anchor_map = torch.cat((grid_x,grid_y,grid_wh),dim=-1)      # shape=(b,3,h,w,4)
            anchor_maps.append(anchor_map.cuda() if self.CUDA else anchor_map)
        return anchor_maps


    def decode_xywh(self,*out_list):
        assert len(out_list) == len(self.base_anchor_maps) , "len(out_list) != len(self.base_anchor_maps)"
        for i in range(len(out_list)):
            out_list[i][...,:2] += self.base_anchor_maps[i][...,:2]
            out_list[i][...,2:4] = self.base_anchor_maps[i][...,2:4]*out_list[i][...,2:4]
            out_list[i][...,:4] *= self.strides[i]
        
        """
        #----------------------------------------------------------------------#
        # 坐标调整到输入图片的尺度
        # out1:   tensor.shape=(b_size , c_size , h_size1 , w_size1 , n_attr)
        # out2:   tensor.shape=(b_size , c_size , h_size2 , w_size2 , n_attr)
        # out3:   tensor.shape=(b_size , c_size , h_size3 , w_size3 , n_attr)
        #----------------------------------------------------------------------#
        """
        return out_list

    def select_every_class_boxes(self,*out_list):
        unfold_out1,unfold_out2,unfold_out3 = out_list
        # unfold_outi : shape=(b_size , num_i , 5+class)
        unfold_out1 = unfold_out1.view(self.batch_size,-1,self.bbox_attrs)
        unfold_out2 = unfold_out2.view(self.batch_size,-1,self.bbox_attrs)
        unfold_out3 = unfold_out3.view(self.batch_size,-1,self.bbox_attrs)
        # unfold_outs : shape=(b_size , num_1+num_2+num_3 , 5+class)
        unfold_outs = torch.cat((unfold_out1,unfold_out2,unfold_out3),dim=1)
        # unfold_outs_class_conf: shape=(b_size,num1+num2+num3,class)
        unfold_outs_class_conf = torch.unsqueeze(unfold_outs[...,4],dim=2)*unfold_outs[...,5:]
        # confs: shape=(b_size , num1+num2+num3)
        # cls_ids: shape=(b_size , num1+num2+num3)
        confs , cls_ids = torch.max(unfold_outs_class_conf,dim=-1)
        # masks: shape=(b_size , num1+num2+num3)
        masks = confs > self.yolo_thre_conf
        boxes_list = []
        confs_list = []
        cls_ids_list = []
        for b_i in range(self.batch_size):
            mask_i = masks[b_i]
            boxes_oneimg = unfold_outs[b_i,:,:4][mask_i]
            if len(boxes_oneimg)>0:
                boxes_list.append(boxes_oneimg)
                confs_list.append(confs[b_i][mask_i])
                cls_ids_list.append(cls_ids[b_i][mask_i])
            else:
                boxes_list.append(torch.zeros(1,4))
                confs_list.append(torch.zeros(1,))
                cls_ids_list.append(torch.zeros(1,))

        return boxes_list , confs_list , cls_ids_list

    def select_person_boxes(self,boxes_list_a_batch , confs_list_a_batch , cls_ids_list_a_batch):
        #------------------------------------------------------------------------------------------#
        # boxes_list_a_batch:    [(new_num , 4),(new_num , 4) , ...]        # 4 = cx , cy , w , h
        # confs_list_a_batch:    [(new_num , ) ,(new_num , ) , ... ]
        # cls_ids_list_a_batch:  [(new_num , ) ,(new_num , ) , ... ]
        # 若一张图中未检测出任何目标框 ， 则(new_num , 4) ， (new_num , ) ，(new_num , )全为零  
        #------------------------------------------------------------------------------------------#
        person_boxes = []
        perosn_confs = []
        for bi in range(self.batch_size):
            boxes = boxes_list_a_batch[bi]          # (new_num , 4)
            confs = confs_list_a_batch[bi]          # (new_num ,  )
            cls_ids = cls_ids_list_a_batch[bi]      # (new_num ,  )
            mask = cls_ids == self.cfg.person_id
            p_boxes = boxes[mask]       # (n,4)
            p_confs = confs[mask]       # (n,)
            # 若一张图中未检测出任何目标框 ，或者该图中没有任何人体
            # 则图像中所有的人体框坐标全为零
            if torch.sum(p_confs)==0:
                person_boxes.append(torch.zeros(1,4))
                perosn_confs.append(torch.zeros(1,))
            else:
                person_boxes.append(p_boxes)
                perosn_confs.append(p_confs)

        return person_boxes , perosn_confs


    def __call__(self,out1,out2,out3):
        b1,c1,h1,w1 = out1.shape
        b2,c2,h2,w2 = out2.shape
        b3,c3,h3,w3 = out3.shape
        #   batch_size, 3, 13, 13, 5 + num_classes
        out1 = out1.view(b1,self.point_bbox,self.bbox_attrs,h1,w1).permute(0,1,3,4,2).contiguous()
        out1[...,:2]  = torch.sigmoid(out1[...,:2])
        out1[...,2:4] = torch.exp(out1[...,2:4])
        out1[...,4:]  = torch.sigmoid(out1[...,4:])
        #   batch_size, 3, 26, 26, 5 + num_classes
        out2 = out2.view(b2,self.point_bbox,self.bbox_attrs,h2,w2).permute(0,1,3,4,2).contiguous()
        out2[...,:2] = torch.sigmoid(out2[...,:2])
        out2[...,2:4] = torch.exp(out2[...,2:4])
        out2[...,4:] = torch.sigmoid(out2[...,4:])
        #   batch_size, 3, 52, 52, 5 + num_classes
        out3 = out3.view(b2,self.point_bbox,self.bbox_attrs,h3,w3).permute(0,1,3,4,2).contiguous()
        out3[...,:2] = torch.sigmoid(out3[...,:2])
        out3[...,2:4] = torch.exp(out3[...,2:4])
        out3[...,4:] = torch.sigmoid(out3[...,4:])

        #----------------------------------------------------------------------#
        # yolo_out:     得到预测框在输入图片上的真实位置xy和wh
        #   out1:   tensor.shape=(b_size , c_size , h_size1 , w_size1 , n_attr=5+nclass)
        #   out2:   tensor.shape=(b_size , c_size , h_size2 , w_size2 , n_attr)
        #   out3:   tensor.shape=(b_size , c_size , h_size3 , w_size3 , n_attr)
        #----------------------------------------------------------------------#
        out1 , out2 , out3 = self.decode_xywh(out1,out2,out3)

        
        if self.training:
            if self.training_yolo:
                return out1 , out2 , out3

        # inference
        else:       
            #------------------------------------------------------------------------------------------#
            # boxes_list_a_batch:    [(new_num , 4),(new_num , 4) , ...]        # 4 = cx , cy , w , h
            # confs_list_a_batch:    [(new_num , ) ,(new_num , ) , ... ]
            # cls_ids_list_a_batch:  [(new_num , ) ,(new_num , ) , ... ]
            # 若一张图中未检测出任何目标框 ， 则(new_num , 4) ， (new_num , ) ，(new_num , )全为零
            #------------------------------------------------------------------------------------------#
            boxes_list_a_batch , confs_list_a_batch , cls_ids_list_a_batch = self.select_every_class_boxes(out1,out2,out3)
            #------------------------------------------------------------------------------------------#
            # person_boxes:     [(num , 4),(num , 4) , ...]             # 4 = cx , cy , w , h
            # perosn_confs:     [(num , ) ,(num , ) , ... ]
            # 若一张图中未检测出任何目标框 ，或者该图中没有任何人体
            # 则图像中所有的人体框坐标全为零torch.zeros(1,4)
            #------------------------------------------------------------------------------------------#
            person_boxes , perosn_confs = self.select_person_boxes(boxes_list_a_batch , confs_list_a_batch , cls_ids_list_a_batch)
            # person_boxes_with_nms:    [(n1 , 4),(n2 , 4) , ...]          # 4 = cx , cy , w , h
            # 如果该图片中没有人体框，则直接令其为 torch.zeros(1,4)
            person_boxes_with_nums = []
            for i , boxes in enumerate(person_boxes):
                # cx , cy , w , h ---> x1,y1,x2,y2
                boxes[:,:2] , boxes[:,2:4] = boxes[:,:2]+boxes[:,2:4]/2 , boxes[:,:2]+boxes[:,2:4]/2
                boxes_with_nms = self.NMS(boxes , perosn_confs[i],self.cfg.nms_iou_thre)
                # x1,y1,x2,y2 ---> cx , cy , w , h
                boxes_with_nms[:,:2] , boxes_with_nms[:,2:4] = (boxes_with_nms[:,:2]+boxes_with_nms[:,2:4])/2 , boxes_with_nms[:,2:4]-boxes_with_nms[:,:2]
                person_boxes_with_nums.append(boxes_with_nms)
                
            return person_boxes_with_nums
