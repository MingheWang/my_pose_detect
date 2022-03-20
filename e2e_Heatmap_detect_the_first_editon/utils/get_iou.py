import torch

def Get_box_iou(box_a,box_b,iou_type='iou'):
    """
    # box_a : tensor_shape=(n,4) x1y1x2y2
    # box_b : tensor_shape=(m,4) x1y1x2y2
    """
    box_a = torch.unsqueeze(box_a,dim=1)    # shape=(n,1,4)

    inter_lt = torch.maximum(box_a[...,:2],box_b[:,:2])   # shape=(n,m,2)
    inter_rb = torch.minimum(box_a[...,2:],box_b[:,2:])   # shape=(n,m,2)
    inter_wh = inter_rb - inter_lt      # shape=(n,m,2)
    inter_area = torch.prod(inter_wh,dim=2)    
    inter_area = torch.clip(inter_area,min=0)   # shape=(n,m)

    box_a_wh = box_a[...,2:] - box_a[...,:2]    # shape=(n,1,2)
    area_a = torch.prod(box_a_wh ,dim=2)        # shape=(n,1)
    box_b_wh = box_b[:,2:] - box_b[:,:2]        # shape=(m,2)
    area_b = torch.prod(box_b_wh , dim=1)       # shape=(m,)
    union_area = area_a + area_b - inter_area   # shape=(n,m)
    if iou_type.lower() == 'iou':
        iou = inter_area/union_area             # shape=(n,m)
    if iou_type.lower()=='ciou':
        pass
    return iou