from csv import writer
import torch
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.config import Cfg
from easydict import EasyDict
from Models.model import get_model
from tools.cocokeypoint_dataset import CocoKeypoint
from tools.detect_dataset import  Detect_Dataset
from utils.config import Cfg
from training.yolo_loss import YoloLoss , originloss
from training.pose_loss import JointsMSELoss
from training.train_on_one_epoch import Train_yolo_one_epoch

def pose_collate_fn(batch):
    images = []
    person_boxes = []
    heatmaps_list = []
    # img: tensor(3,h,w)
    # box: tensor(n_person,4)                 # (x1,y1,w,h)
    # heatmaps.shape = (n_person,17,H,W)      # H,W = output_heatmap_size
    for img,box,heatmap in batch:
        images.append(img)
        person_boxes.append(box)
        heatmaps_list.append(heatmap)
    images = torch.stack(images,dim=0)

    return images , person_boxes , heatmaps_list

def detect_dataset_collate(batch):
    images = []
    bboxes = []
    # img: tensor.shape=(3,image_size,image_size)
    # box: tensor.shape=(n,5)        # 5 = x1,y1,w,h,id
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.stack(images,dim=0)
    return images, bboxes

def new_detect_collate(batch):
    images = []
    bboxes = []
    # img: tensor.shape=(3,image_size,image_size)
    # box: tensor.shape=(n,5)        # 5 = x1,y1,x2,y2,id
    for img,box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.stack(images,dim=0)
    return images , bboxes




def train(model,train_pose_dataloader,cfg,device,writer_dict):
    
    # if cfg.training_yolo:
    #     for param in model.backbone.parameters():
    #         param.requires_grid = True
    #     for param in model.detect_net.parameters():
    #         param.requires_grid = True

    #     anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #     anchors = np.array([12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]).reshape(-1,2)
    #     num_classes = 80
    #     input_shape=[416,416]
    #     Cuda = True
    #     label_smothing=True
    #     Yolo_loss = originloss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    #     detect_optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay = 5e-4)
    #     if cfg.Cosine_lr:
    #         lr_scheduler_detect = optim.lr_scheduler.CosineAnnealingLR(detect_optimizer, T_max=5, eta_min=1e-5)
    #     else:
    #         lr_scheduler_detect = optim.lr_scheduler.StepLR(detect_optimizer, step_size=1, gamma=0.94)
    #     epoch_steps_train = len(train_detect_dataloader.dataset) // cfg.batch_size
    #     # epoch_steps_val = len(val_detect_dataloader.dataset) // cfg.batch_size
    #     # epoch_steps_val = len(val_detect_dataloader.dataset) // cfg.batch_size
    #     num_epochs = cfg.TRAIN_YOLO_EPOCHS
        
    #     for epoch in range(num_epochs):
    #         model.train()
    #         train_loss = 0
    #         with tqdm(total=epoch_steps_train,ncols=100,desc=f"train:{epoch+1}/{num_epochs}",postfix=dict) as pbar:
    #             for i , (images , boxes ) in enumerate(train_detect_dataloader):
                    
    #                 images = images.type(torch.FloatTensor).to(device)
    #                 boxes = [box.type(torch.FloatTensor).to(device) for box in boxes]
                    
    #                 yolo_outs = model(images)
                    
    #                 # loss = Yolo_loss(yolo_outs,boxes)
    #                 loss_value_all  = 0
    #                 num_pos_all     = 0
    #                 for l in range(len(yolo_outs)):
    #                     loss_item, num_pos = Yolo_loss(l, yolo_outs[l], boxes)
    #                     loss_value_all  += loss_item
    #                     num_pos_all     += num_pos
    #                 loss_value = loss_value_all / num_pos_all
                    
    #                 detect_optimizer.zero_grad()
    #                 loss_value.backward()
    #                 detect_optimizer.step()

    #                 train_loss +=loss_value.item()
    #                 del images, boxes, yolo_outs, loss_value
    #                 torch.cuda.empty_cache()

    #                 if i % cfg.PRINT_FREQ == 0:
    #                     writer = writer_dict['writer']
    #                     train_detect_global_steps = writer_dict['train_detect_global_steps']
    #                     writer.add_scalar('train_detect_loss', loss_value.item(), train_detect_global_steps)
    #                     writer_dict['train_detect_global_steps'] = train_detect_global_steps + 1

    #                 pbar.set_postfix(**{"averg_loss":train_loss/(i+1)})
    #                 pbar.update(1)
                    

                    
    #         lr_scheduler_detect.step()

    #         val_loss = 0
    #         with torch.no_grad():
    #             model.eval()
    #             with tqdm(total=epoch_steps_val,desc=f"val:{epoch-1}/{num_epochs}" , ncols=100, postfix=dict) as pbar:
    #                 for j , (images , boxes ) in enumerate(val_detect_dataloader):
    #                     yolo_outs = model(images)
    #                     loss = Yolo_loss(yolo_outs,boxes)
    #                     if i % cfg.PRINT_FREQ == 0:
    #                         writer = writer_dict['writer']
    #                         val_detect_global_steps = writer_dict['val_detect_global_steps']
    #                         writer.add_scalar('val_detect_loss', loss.item(), val_detect_global_steps)
    #                         writer_dict['val_detect_global_steps'] = val_detect_global_steps + 1
    #                     val_loss += loss
    #                     pbar.set_postfix(**{"detect_loss":val_loss/(j+1)})
    #                     pbar.update(1)
            
                    
    if cfg.training_pose:
        for param in model.backbone.parameters():
            param.requires_grid = True
        for param in model.posenet.parameters():
            param.requires_grid = True
        
        Pose_loss = JointsMSELoss().to(device)

        pose_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 5e-4)
        if cfg.Cosine_lr:
            lr_scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler_pose = optim.lr_scheduler.StepLR(pose_optimizer, step_size=1, gamma=0.94)
        epoch_steps_train = len(train_pose_dataloader.dataset) // (cfg.batch_size//2)
        # epoch_steps_val = len(val_pose_dataloader.dataset) // (cfg.batch_size//2)
        num_epochs = cfg.TRAIN_POSE_EPOCHS
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            with tqdm(total=epoch_steps_train,ncols=100,desc=f"train:{epoch+1}/{num_epochs}",postfix=dict) as pbar:
                for i , (images , gt_boxes , gt_heatmaps) in enumerate(train_pose_dataloader):
                    images = images.type(torch.FloatTensor).to(device)
                    gt_boxes = [box.type(torch.FloatTensor).to(device) for box in gt_boxes]
                    gt_heatmaps = [heatmap.type(torch.FloatTensor).to(device) for heatmap in gt_heatmaps]
                    pose_outs , gt_heatmaps = model(images,gt_boxes,gt_heatmaps)
                    loss = Pose_loss(pose_outs,gt_heatmaps)
                    pose_optimizer.zero_grad()
                    loss.backward()
                    pose_optimizer.step()
                    if i % cfg.PRINT_FREQ == 0:
                        writer = writer_dict['writer']
                        train_pose_global_steps = writer_dict['train_pose_global_steps']
                        writer.add_scalar('train_pose_loss', loss.item(), train_pose_global_steps)
                        writer_dict['train_pose_global_steps'] = train_pose_global_steps + 1
                    pbar.set_postfix(**{"pose_loss":(train_loss+loss.item())/(i+1)})
                    pbar.update(1)
            lr_scheduler_pose.step()
            
            # val_loss = 0
            # with torch.no_grad():
            #     model.eval()
            #     with tqdm(total=epoch_steps_val,desc=f"val:{epoch-1}/{num_epochs}" , ncols=100, postfix=dict) as pbar:
            #         for j , (images , boxes , heatmaps) in enumerate(val_pose_dataloader):
            #             pose_outs = model(images,boxes)
            #             loss = Pose_loss(pose_outs,boxes,heatmaps)
            #             val_loss += loss
            #             if i % cfg.PRINT_FREQ == 0:
            #                 writer = writer_dict['writer']
            #                 val_pose_global_steps = writer_dict['train_pose_global_steps']
            #                 writer.add_scalar('val_pose_loss', loss.item(), val_pose_global_steps)
            #                 writer_dict['val_pose_global_steps'] = val_pose_global_steps + 1
            #             pbar.set_postfix(**{"aver_loss":val_loss/(j+1)})
            #             pbar.update(1)

    writer_dict["writer"].close()
            


def main():
    cfg = parse_args(**Cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)
    
    
    if cfg.weight_path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.weight_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # train_detect_dataset = Detect_Dataset(cfg.root, cfg.train_detect_anno_name, cfg.image_size, cfg.use_mosaic, is_train=True)
    # train_detect_dataloader = DataLoader(dataset=train_detect_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=0,pin_memory=True,collate_fn=detect_dataset_collate)

    # new_train_detect_dataset = Detect_Dataset(cfg.root, cfg.train_detect_anno_name, cfg.image_size, cfg.use_mosaic, is_train=True)
    # new_train_detect_dataloader = DataLoader(dataset=new_train_detect_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=0,pin_memory=True,collate_fn=new_detect_collate)

    # val_detect_dataset = Detect_Dataset(cfg.root, cfg.val_detect_anno_name, cfg.image_size)
    # val_detect_dataloader = DataLoader(dataset=train_detect_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=4,pin_memory=True,collate_fn=detect_dataset_collate)

    train_pose_dataset = CocoKeypoint(cfg,cfg.root,cfg.train_pose_anno_name)
    train_pose_dataloader = DataLoader(dataset=train_pose_dataset,batch_size=cfg.batch_size//2,shuffle=True,num_workers=0,pin_memory=True,collate_fn=pose_collate_fn)

    # val_pose_dataset = CocoKeypoint(cfg,cfg.root,cfg.val_pose_anno_name)
    # val_pose_dataloader = DataLoader(dataset=val_pose_dataset,batch_size=cfg.batch_size//2,shuffle=True,num_workers=4,pin_memory=True,collate_fn=pose_collate_fn)

    writer_dict =  {
                    'writer': SummaryWriter(log_dir=cfg.tb_log_dir),
                    'train_detect_global_steps': 0,
                    'valid_detect_global_steps': 0,
                    'train_pose_global_steps': 0,
                    'train_pose_global_steps': 0,
                    }
    # writer_dict["writer"].add_graph(model,(torch.ones(cfg.batch_size,3,cfg.image_size,cfg.image_size)))

    # train(model,train_detect_dataloader,val_detect_dataloader,train_pose_dataloader,val_pose_dataloader,cfg,device,writer_dict)
    # train(model,new_train_detect_dataloader,cfg,device,writer_dict)
    train(model,train_pose_dataloader,cfg,device,writer_dict)



def parse_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="dataset dir",default="/home/wmh/Datasets/coco2017")
    parser.add_argument("--out_dir",help="save path dir",default="../out_file")
    parser.add_argument('--iou_type', type=str, default='iou',help='iou type (iou, giou, diou, ciou)')
    parser.add_argument("--training_yolo", help="train_yolo only" , default=False)
    parser.add_argument("--training_pose", help="train_pose only" , default=True)
    
    args = vars(parser.parse_args())
    
    cfg.update(args)

    return EasyDict(cfg)

if __name__ == "__main__":
    
    main()
