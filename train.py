from ultralytics import YOLO, RTDETR 

# Load a model
# model = YOLO("ultralytics/cfg/models/11/yolo11s-eafpn.yaml")
# model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")

# model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml")
# model = YOLO("ultralytics/cfg/models/v8/yolov8n-p2-repvgg-sf.yaml")
# model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml")
model = YOLO("ultralytics/cfg/models/v10/yolov10s.yaml")
# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=1024,device=[0,1,2,3],batch=8,project="runs/detect/visdrone/yolov10s",pretrained=False)

