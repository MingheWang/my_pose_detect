from easydict import EasyDict

Cfg = EasyDict()

Cfg.tb_log_dir = "./log"
Cfg.PRINT_FREQ = 30

Cfg.CUDA = True

Cfg.image_size = 416
Cfg.batch_size = 8
Cfg.detect_bsize = Cfg.batch_size
Cfg.pose_bsize = Cfg.batch_size //2
Cfg.lr = 1e-3

Cfg.weight_path = ""

Cfg.root = "/home/wmh/Datasets/coco2017"

Cfg.train_detect_anno_name = "instances_train2017"
Cfg.val_detect_anno_name = "instances_val2017"
Cfg.num_classes = 80
Cfg.point_bbox = 3
Cfg.out_dir = "../out_file"
Cfg.use_mosaic = False
Cfg.TRAIN_YOLO_EPOCHS = 300
Cfg.yolo_thre_with_obj = 0.5
Cfg.yolo_ignore_thre = 0.3
Cfg.yolo_bk_layers = [1, 2, 4, 4, 3]
Cfg.iou_type = "iou"
Cfg.anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
Cfg.strides = [8,16,32]
Cfg.num_classes = 80
Cfg.bbox_attrs = 5 + Cfg.num_classes
Cfg.Cosine_lr = True
Cfg.yolo_thre_conf = 0.5
Cfg.nms_iou_thre = 0.5

Cfg.TRAIN_POSE_EPOCHS = 300
Cfg.train_pose_anno_name = "person_keypoints_train2017"
Cfg.val_pose_anno_name = "person_keypoints_val2017"

Cfg.posnet_input_dim = 256+512
Cfg.nstack = 2
Cfg.heatmap_size = 56
Cfg.num_joints = 17
Cfg.base_sigma = 2
Cfg.with_scale_sigma = False

Cfg.training = True
Cfg.training_yolo = True
Cfg.training_pose = False

Cfg.OPTIMIZER = "adam"




Cfg.hw_ratio = 1.5
Cfg.wh_ratio = 1 / Cfg.hw_ratio








# if Cfg.training_yolo & Cfg.training_pose == True:
#     Cfg.training_together = True
