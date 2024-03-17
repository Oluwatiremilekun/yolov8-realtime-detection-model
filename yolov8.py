import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from ultralytics import YOLO

# Load a pretrained YOLOV8n model

model = YOLO('yolov8m.pt')

# Rin inference on the source
results = model.track(source='video.mp4', show=True, tracker='bytetrack.yaml')
#results = model.track(source=0, show=True, tracker='bytetrack.yaml')