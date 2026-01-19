from ultralytics import YOLO
from ultralytics import RTDETR

if __name__=="__main__":
    # Load a model
    model = YOLO("runs/detect/hazydet/yolo11s-da/train/weights/best.pt")  # pretrained YOLO11n model
    results = model(["../HazyDet/images/test/012003.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="da-yolo_012003.jpg")  # save to disk

    model = YOLO("runs/detect/hazydet/yolo11s/train2/weights/best.pt")  # pretrained YOLO11n model
    results = model(["../HazyDet/images/test/012003.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="yolo11_012003.jpg")  # save to disk
