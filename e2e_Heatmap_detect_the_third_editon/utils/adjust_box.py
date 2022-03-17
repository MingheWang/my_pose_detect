import torch

def adjust_wh_ratio(person_boxes,wh_ratio):
    """
    #---------------------------------------------------------------------------------#
    # arg:
    #    person_boxes:   [(n1 , 4),(n2 , 4) , ...]          # 4 = cx , cy , w , h
    #    wh_ratio:   float
    # return:
    #    new_person_boxes:  [(n1 , 4),(n2 , 4) , ...]          # 4 = cx , cy , w , h
    #---------------------------------------------------------------------------------#
    """
    new_person_boxes = []

    for boxes in person_boxes:
        # boxes: shape=(n,4)
        w2h = boxes[:,2]/boxes[:,3]     # shape=(n,)
        mask = w2h < wh_ratio
        boxes1 = boxes[mask]
        boxes1[:,2] = boxes1[:,3] * wh_ratio
        boxes2 = boxes[~mask]
        boxes2[:,3] = boxes2[:,2] / wh_ratio
        new_boxes = torch.cat((boxes1,boxes2),dim=0)

    new_person_boxes.append(new_boxes)

    return new_person_boxes
