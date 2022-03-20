
from tqdm import tqdm

def Train_yolo_one_epoch(epoch,model,dataloader,epoch_steps,num_epochs , YoloLoss , optimizer):
    with tqdm(total=epoch_steps,ncols=100,desc=f"{epoch}/{num_epochs-1}") as pbar:
        for images , boxes , joints in dataloader:
            # yolo_outs: [(b,3,h,w,5+nclass),(b,3,h,w,5+nclass),(b,3,h,w,5+nclass)]
            yolo_outs = model(images)
            loss = YoloLoss(yolo_outs,boxes)
            optimizer.zero_grad()
            loss.backward()

            pbar.update(1)
            
