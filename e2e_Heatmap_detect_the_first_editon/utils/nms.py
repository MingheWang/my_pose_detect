from operator import index
import torch

def nms_fn(person_boxes,person_confs,iou_thre):
    """
    #----------------------------------------------------------------------------------#
    # arg:
    #   person_boxes:    [(num , 4),(num , 4) , ...]             # 4 = cx , cy , w , h
    #   person_confs:    [(num , ) ,(num , ) , ... ]
    #   iou_thre: a
    #----------------------------------------------------------------------------------#
    # return:
        person_boxes:    [(n1 , 4),(n2 , 4) , ...]          # 4 = cx , cy , w , h
    #----------------------------------------------------------------------------------#
    """
    person_boxes = []
    batch_size = len(person_boxes)
    for boxes , confs in zip(person_boxes,person_confs):
        # 一张图片
        if torch.sum(confs)==0:
            # 如果该图片中没有人体框，则直接令其为torch.zeros(1,4)
            person_boxes.append(torch.zeros(1,4))
            continue
        confs , index = torch.sort(confs,descending=True)
        boxes = boxes[index]
        person_boxes_aimg = []
        for i in range(len(confs)):
            # 逐一访问该图中的每个人体框
            person_boxes_aimg.append(boxes[0])      # person_boxes_aimg.shape = (4,)
            if len(boxes)==1:
                break
            boxes = boxes[1:]
            # temp = []
            # for box in boxes:
            #     iou = iou_fn(box,person_boxes_aimg[i])
            #     if iou < iou_thre:
            #         temp.append(box)
            # boxes = temp
            ious = iou_fn(person_boxes[i],boxes)
            mask = ious < iou_thre      # shape=(k,)
            boxes = boxes[mask]
        person_boxes.append(torch.stack(person_boxes_aimg,dim=0))


def iou_fn(box1,boxes2):
    """
    args:
        box1: shape=(4,)        # 4 = cx , cy , w , h
        boxes: shape=(n,4)      # 4 = cx , cy , w , h
    """
    area1 = torch.prod(box1[2:4])       # shape=(1,)
    box1[:2], box1[2:4] = box1[:2]-box1[2:4]/2 , box1[:2]+box1[2:4]

    area2 = torch.prod(boxes2[:,2:4],dim=1)     # shape=(n,)
    boxes2[:,:2], boxes2[:,2:4] = boxes2[:,:2]-boxes2[:,2:4]/2 , boxes2[:,:2]+boxes2[:,2:4]/2
    # lt: shape=(n,2)
    lt = torch.max(box1[:2],boxes2[:,:2])
    # rb: shape=(n,2)
    rb = torch.min(box1[2:4],boxes2[:,2:4])
    union_wh = torch.clamp(rb-lt , min=0)
    union_area = torch.prod(union_wh,dim=1)     # shape=(n,)
    ious = union_area/(area1+area2-union_area)   # shape=(n,)

    return ious

