#Training on custom Road sign detection dataset
from ultralytics import YOLO

# Load the base model (you can choose yolov8n.pt, yolov8s.pt, etc.)
model = YOLO('yolo12n.pt')

# Fine-tune on custom dataset
model.train(
    data='processed_yolo_dataset/data.yaml',  # Your custom dataset YAML path
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.001,             # initial learning rate
    optimizer='SGD',       # 'SGD', 'Adam', or 'AdamW'
    momentum=0.9,          # used with SGD
    weight_decay=0.0005,   # L2 regularization
    warmup_epochs=3,       # number of warmup epochs
    workers=2              # number of dataloader workers
)

#
