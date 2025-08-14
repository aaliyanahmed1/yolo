🚦 Road Sign Detection using YOLOv12n
A custom-trained road sign detection model based on YOLOv12n, fine-tuned on a custom-labeled dataset of road signs. This project demonstrates how to efficiently detect various traffic signs using a lightweight and real-time-capable model.

📌 Overview
This project involves:

Preparing a dataset in YOLO format for road sign detection

Fine-tuning the YOLOv12n model

Validating the model on a separate validation set

Performing test inference on real-world images

Saving results and performance metrics for review

🧠 Tech Stack
YOLOv12n – for lightweight, high-speed object detection

Python 3.8+

Ultralytics – model training and evaluation framework

OpenCV (optional) – for visualization and image handling

CUDA (optional) – for GPU acceleration


Training Parameters:

Epochs: 100

Batch Size: 16

Image Size: 640×640

Optimizer: SGD

Learning Rate: 0.001

Trained model weights and logs will be saved in:

bash
Copy
Edit
runs/train/road_sign_yolo12n/
✅ Validation
Validation is performed automatically after training and can be re-run using:

python
Copy
Edit
model.val(data='processed_yolo_dataset/data.yaml', imgsz=640, split='val')
Metrics like Precision, Recall, mAP50, and mAP50-95 are computed.

🧪 Testing / Inference
To test on a new image:

python
Copy
Edit
model.predict(source='test_images/sample_road_sign.jpg', conf=0.25, save=True)
This will save the prediction image with bounding boxes and detected class labels.

📤 Export (Optional)
To export the trained model for deployment:

python
Copy
Edit
model.export(format='onnx')     # or 'torchscript', 'engine' etc.
📈 Sample Results
Metric	Value
Precision	0.92
Recall	0.89
mAP@0.5	0.94
mAP@0.5:0.95	0.79

Note: Results will vary depending on dataset quality and augmentation.
