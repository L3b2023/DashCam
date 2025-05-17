from ultralytics import YOLO
import cv2
import numpy as np
import random
import time

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
    if label in CLASS_COLOR_MAP:
        return CLASS_COLOR_MAP[label]
    else:
        random.seed(hash(label))
        return tuple(random.randint(50, 255) for _ in range(3))

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

def draw_text_with_outline(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255, 255, 255), thickness=1):
    x, y = pos
    # Contorno preto
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Texto branco
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame")
        break

    results = model.predict(source=frame, conf=0.3, verbose=False, stream=False)
    annotated = frame.copy()

    # Contadores
    num_carros = 0
    num_pedestres = 0
    num_sinais = 0

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

            # Contar classes específicas
            if label in ['car', 'truck', 'bicycle', 'motorcycle']:
                num_carros += 1
            elif label == 'person':
                num_pedestres += 1
            elif label in ['traffic light', 'stop sign']:
                num_sinais += 1

            if label == 'traffic light':
                semaforo_cor = detect_traffic_light_color(frame, (x1, y1, x2, y2))
                if semaforo_cor:
                    label_text += f" ({semaforo_cor.upper()})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            text = label_text
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_pos = (x1, y1 - 5 if y1 - 5 > 10 else y1 + text_size[1] + 5)

            cv2.putText(annotated, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # Calcular FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Exibir HUD no canto superior esquerdo
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Carros: {num_carros}",
        f"Pedestres: {num_pedestres}",
        f"Sinais: {num_sinais}"
    ]
    for i, line in enumerate(hud_lines):
        draw_text_with_outline(annotated, line, (10, 20 + i * 20))

    cv2.imshow("testing", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
