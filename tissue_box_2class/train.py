# pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # small and fast; use yolov8s.pt for higher accuracy
model.train(data="D:\AI(Dr.Ali trainning)\Graduation Project\DEMO_3\tissue_box_2class\tissue_box_2class\Tissue Defect Detection.v1-tissue-dataset.yolov11\data.yaml", epochs=50, imgsz=640, batch=16)
metrics = model.val(split="test")    # final evaluation on the test split
print("mAP50:", metrics.box.map50)
