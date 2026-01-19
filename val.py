from ultralytics import YOLO
from ultralytics import RTDETR

if __name__=="__main__":
    # model = RTDETR('runs/detect/rt_detr_l/train/weights/best.pt')
    
    model = YOLO('runs/detect/visdrone/yolov10s/train/weights/best.pt')
    results = model.val(split="test",data="VisDrone.yaml",imgsz=1024,batch=16, project="runs/detect/visdrone/yolov10s/")

