from ultralytics import YOLO
import cv2
import numpy as np
import random
import time
from datetime import datetime

model = YOLO("yolov8n.pt")

CLASS_COLOR_MAP = {
    'person':        (255, 0, 0),
    'car':           (0, 255, 0),
    'truck':         (0, 0, 255),
    'traffic light': (0, 255, 255),
    'stop sign':     (255, 255, 0),
    'bicycle':       (255, 0, 255),
    'motorcycle':    (0, 165, 255),
}

def get_color_for_label(label):
    return CLASS_COLOR_MAP.get(label, (255, 255, 255))

def detect_traffic_light_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        'red': cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        ),
        'green': cv2.inRange(hsv, (40, 70, 50), (90, 255, 255)),
    }
    counts = {color: cv2.countNonZero(mask) for color, mask in masks.items()}
    detected = max(counts, key=counts.get)
    return detected if counts[detected] > 10 else None

def draw_text(img, text, pos, color=(255,255,255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    x, y = pos
    # Contorno preto leve
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    # Texto branco
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# Sessão
session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_log = f"session_{session_start}.txt"

# Estatísticas acumuladas
total_vehicles = 0
total_pedestrians = 0
total_signs = 0
total_cars = 0
total_trucks = 0
total_motorcycles = 0
fps_values = []

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(source=frame, conf=0.3, verbose=False, stream=False)
    annotated = frame.copy()

    # Contadores por frame
    frame_cars = 0
    frame_trucks = 0
    frame_motorcycles = 0
    frame_pedestrians = 0
    frame_signs = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            score = float(box.conf[0])
            if score < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = get_color_for_label(label)
            label_text = label

            if label == 'car':
                frame_cars += 1
            elif label == 'truck':
                frame_trucks += 1
            elif label == 'motorcycle':
                frame_motorcycles += 1
            elif label == 'person':
                frame_pedestrians += 1
            elif label in ['traffic light', 'stop sign']:
                frame_signs += 1

            if label == 'traffic light':
                light_color = detect_traffic_light_color(frame, (x1, y1, x2, y2))
                if light_color:
                    label_text += f" ({light_color.upper()})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            draw_text(annotated, label_text, (x1, y1 - 5))

    # Acumular totais
    total_cars += frame_cars
    total_trucks += frame_trucks
    total_motorcycles += frame_motorcycles
    total_pedestrians += frame_pedestrians
    total_signs += frame_signs
    total_vehicles += frame_cars + frame_trucks + frame_motorcycles

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_values.append(fps)
    fps_avg = sum(fps_values) / len(fps_values)
    fps_max = max(fps_values)
    fps_min = min(fps_values)

    # HUD
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Vehicles: {frame_cars + frame_trucks + frame_motorcycles}",
        f"Pedestrians: {frame_pedestrians}",
        f"Traffic Signs: {frame_signs}",
        f"FPS Max: {fps_max:.1f}",
        f"FPS Min: {fps_min:.1f}",
        f"FPS Avg: {fps_avg:.1f}"
    ]
    for i, line in enumerate(hud_lines):
        draw_text(annotated, line, (10, 20 + i * 18))

    cv2.imshow("Object Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()

# Salvar estatísticas da sessão
with open(session_log, "w") as f:
    f.write(f"Session started at: {session_start}\n\n")
    f.write("Total objects detected:\n")
    f.write(f"Cars: {total_cars}\n")
    f.write(f"Trucks: {total_trucks}\n")
    f.write(f"Motorcycles: {total_motorcycles}\n")
    f.write(f"Pedestrians: {total_pedestrians}\n")
    f.write(f"Traffic Signs: {total_signs}\n\n")
    f.write("FPS Statistics:\n")
    f.write(f"Max FPS: {fps_max:.2f}\n")
    f.write(f"Min FPS: {fps_min:.2f}\n")
    f.write(f"Avg FPS: {fps_avg:.2f}\n")

print(f"Session saved to {session_log}")
