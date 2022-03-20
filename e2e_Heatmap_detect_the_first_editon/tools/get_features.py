import torch

# 得到预测框 、真实框 、 joints 相对于包围框的位置
def get_relative_boxes_and_joint_of_one_image(predict_boxes , truth_boxes , joints):
    """
    ##############################################################################
    Onely do on one image , because each image has difference number of person !!!
    ##############################################################################
    args:
        predict_boxes:  shape = (num_person,4)       $ 4 = x1 , y1 , x2 , y2
        truth_boxes:    shape = (num_person,4)       $ 4 = x1 , y1 , x2 , y2
        joints:         shape = (num_person,17,4)    $ 4 = x , y , vis , sigma   
    return:
        得到预测框 、真实框 、 joints 相对于包围框的位置 
        bounding_boxes:             shape = (num_person,4)      # 4 = x1 , y1 , x2 , y2
        predcit_lt_to_bounding:     shape = (num_person,2)      # 2 = x , y
        joints:                     shape = (num_person,17,4)   # 4 = new_x , new_y , vis , sigma     
    """

    bounding_boxes = torch.zeros(predict_boxes.shape)       # 4 = x1 , y1 , x2 , y2
    bounding_boxes[:,:2] = torch.minimum(predict_boxes[:,:2] , truth_boxes[:,:2])
    bounding_boxes[:,2:] = torch.maximum(predict_boxes[:,2:] , truth_boxes[:,2:])
    
    # 计算predict_boxes左上角相对于bound_box左上角的坐标
    predcit_lt_to_bounding = predict_boxes[:,:2] - bounding_boxes[:,:2]     # shape = (num_person , 2)      2 = x , y

    bounding_boxes = torch.unsqueeze(bounding_boxes,dim=1)      # shape = (num_person,1,4)
    # 计算joints坐标相对于bound_box左上角的坐标
    joints = joints[...,:2] - bounding_boxes[...,:2]            # 4 = new_x , new_y , vis , sigma

    return torch.unsqueeze(bounding_boxes) , predcit_lt_to_bounding , joints

##############################################################################################################################
##############################################################################################################################

# 将预测框调整到需要的高宽比
def adjust_predict_person_boxes_with_a_gaven_scale(predict_boxes_on_a_batch , hw_ratio):
    """
    args:
        predict_boxes_on_a_batch:
            [tensor(num_person1,4) , (num_person2,4) , ....]  ,  len = b_size;      ## 4 = c_x , c_y , w , h
        hw_ratio:
            float;
    return:
        predict_person_boxes_adjusted:
            [tensor(num_person1,4) , (num_person2,4) , ....]  ,  len = b_size;      $$ 4 = x1 , y1 , x2 , y2
            如果该图像中没有人体,则tensor(num_personi,4) = torch.zeros(1,4)
    """
    person_boxes_with_a_batch = []
    for person_boxes in predict_boxes_on_a_batch:
        if torch.sum(person_boxes) == 0:
            person_boxes_with_a_batch.append(torch.zeros(1,4))
            continue
        boxes_h_vs_w = person_boxes[:,3] / person_boxes[:,2]
        mask = boxes_h_vs_w <= hw_ratio
        person_boxes[mask,3] = person_boxes[mask,2] * hw_ratio
        person_boxes[~mask,2] = person_boxes[~mask,3] / hw_ratio
        person_boxes_with_a_batch.append(person_boxes)
    
    predict_boxes_on_a_batch = []
    # (c_x,c_y,w,h) ---> (x1,y1,x2,y2)
    for person_boxes in person_boxes_with_a_batch:
        person_boxes = torch.stack((person_boxes[:,0]-person_boxes[:,2]/2,person_boxes[:,1]-person_boxes[:,3]/2,person_boxes[:,0]+person_boxes[:,2]/2,person_boxes[:,1]+person_boxes[:,3]/2),dim=1)
        predict_boxes_on_a_batch.append(person_boxes)
        
    return predict_boxes_on_a_batch


# 对一张图片的三个特征图进行截切
def cut_feature_on_three_fmap(predict_boxes,feature_map_list,stride_list=None,posenet_input_size=None):
    """
    args:
        predict_boxes:      shape = (num_person,4)      # 4 = x1 , y1 , x2 , y2
        feature_map_list:   [shape=(channel_1,f_size_1,f_size_1) , shape=(channel_2,f_size_2,f_size_2) , shape=(channel_2,f_size_3,f_size_3)]
        stride_list:        [a,b,c]
        posenet_input_size: p_size
    return:
        一张图片上的每个人体框都分别在三个尺度的特征图上截取特征
        cuted_fmap_list:    [[(c1,h,w),(c2,h,w),(c3,h,w)] , [(c1,h,w),(c2,h,w),(c3,h,w)] , [] , ...]      # len = num_person
    """
    cuted_fmap_list = []
    for pred_box in predict_boxes:
        three_cuted_map = []
        pred_box = pred_box/4
        cuted_14_fmap = feature_map_list[0][...,pred_box[1]:pred_box[3]+1,pred_box[2]:pred_box[0]+1]
        three_cuted_map.append(cuted_14_fmap) 
        pred_box = pred_box/2
        cuted_18_fmap = feature_map_list[1][...,pred_box[1]:pred_box[3]+1,pred_box[2]:pred_box[0]+1]
        three_cuted_map.append(cuted_18_fmap)
        pred_box = pred_box/2
        cuted_116_fmap = feature_map_list[1][...,pred_box[1]:pred_box[3]+1,pred_box[2]:pred_box[0]+1]
        three_cuted_map.append(cuted_116_fmap)
        cuted_fmap_list.append(three_cuted_map)

    return cuted_fmap_list



"""
#############################################################################################
# 不能使用一下方法, 因为训练posenet的时候, backward无法传回到backbone层
#############################################################################################
"""
# 获取一个batch_size中的所有图片中的人体的特征图
def cut_feature_map_of_a_batch_image(backbone_output_list_on_a_batch,predict_boxes_on_a_batch,hw_ratio):
    """
    ====================================================================================================================
    args:
        backbone_output_list_on_a_batch:    
            [
             tensor(b_size , c_size , f_size , f_size),
             tensor(b_size , c_size , f_size , f_size),
             tensor(b_size , c_size , f_size , f_size)
            ]

        predict_boxes_on_a_batch:                   #  len = b_size     #  4 = c_x , c_y , w , h
            [
                tensor(num_person1 , 4) , 
                tensor(num_person2 , 4) , 
                ....
            ]                                       

    hw_ratio:
        a_number
    ======================================================================================================================
    return:
        cuted_person_fmap_on_a_batch:   共有batch_size张图片 , 一张图片上的每个人体框都分别在三个尺度的特征图上截取特征
            [[[(c1,h,w),(c2,h,w),(c3,h,w)] , [(c1,h,w),(c2,h,w),(c3,h,w)] , [] , ...],
            [[(c1,h,w),(c2,h,w),(c3,h,w)] , [(c1,h,w),(c2,h,w),(c3,h,w)] , [] , ...],
            ....
            ]           # len = batch_size
        
        predict_boxes_on_a_batch:                   # 4 = x1 , y1 , x2 , y2
            [
                tensor(num_person1 , 4) , 
                tensor(num_person2 , 4) , 
                ....
            ] 
    ======================================================================================================================
    """
    # 调整预测框宽高比 ，且(c_x,c_y,w,h) ---> (x1,y1,x2,y2)
    predict_boxes_on_a_batch = adjust_predict_person_boxes_with_a_gaven_scale(predict_boxes_on_a_batch , hw_ratio)
    cuted_person_fmap_on_a_batch = []
    b_size = backbone_output_list_on_a_batch[0].shape[0]
    for b_idx in range(b_size):
        # [tensor(c_size,f_size,f_size),tensor(c_size,f_size,f_size),tensor(c_size,f_size,f_size)]
        fmap_list_one_image = [backbone_output_list_on_a_batch[i][b_idx] for i in range(len(backbone_output_list_on_a_batch))]
        # tensor(num_person,4)
        predict_boxes_one_image = predict_boxes_on_a_batch[b_idx]
        # cuted_fmap_list:    [[(c1,h,w),(c2,h,w),(c3,h,w)] , [(c1,h,w),(c2,h,w),(c3,h,w)] , [] , ...]      # len = num_person
        cuted_fmap_list = cut_feature_on_three_fmap(predict_boxes=predict_boxes_one_image,feature_map_list=fmap_list_one_image)
        cuted_person_fmap_on_a_batch.append(cuted_fmap_list)
    
    #----------------------------------------------------------------------------------------------------------------------------#
    # predict_boxes_on_a_batch：
    #           [(n1 , 4),(n2 , 4) , ...]          # 4 = x1 , y1 , x2 , y2
    # cuted_person_fmap_on_a_batch:
    #           list列表，每个元素对应一张图片，
    #           [[(c1,h,w),(c2,h,w),(c3,h,w)] , [(c1,h,w),(c2,h,w),(c3,h,w)] , [] , ...]      # len = num_person一张图片中人的数目
    #----------------------------------------------------------------------------------------------------------------------------#
    return cuted_person_fmap_on_a_batch , predict_boxes_on_a_batch


