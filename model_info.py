from ultralytics import YOLO
# 加载训练好的模型或者网络结构配置文件
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n-da.yaml')
    print(model.info())
