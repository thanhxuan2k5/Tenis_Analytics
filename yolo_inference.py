from ultralytics import YOLO
model = YOLO('yolov8n')

result = model.predict(r'D:/Tenis_Analysts/input/image.png', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)