import cv2
import torch
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)  # 0 = webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tahmin yap
    predictions = model(frame)

    # Sonuçları görselleştir
    annotated_frame = predictions[0].plot()

    # Göster
    cv2.imshow('YOLOv8 - Webcam', annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()