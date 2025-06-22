import sys
import os
import cv2
import torch
import numpy as np
import time
from models.common import DetectMultiBackend
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Initialize YOLOv5 model with lighter Nano version and GPU support if available
weights_path = os.path.join('weights', 'yolov5n.pt')  # Changed to yolov5n.pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DetectMultiBackend(weights_path, device=device)
model.eval()

# Get class names
CLASS_NAMES = model.names if hasattr(model, 'names') else {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Initialize DeepSort tracker with optimized parameters
deep_sort = DeepSort(max_age=5, n_init=1, max_iou_distance=0.7)  # Reduced max_age for faster updates

# Start video capture from file
video_path = r'D:\Telegram Desktop\56310-479197605_tiny.mp4'
cap = cv2.VideoCapture(video_path)

# Class-wise vehicle count
vehicle_counts = defaultdict(int)
already_counted_ids = set()

# Resize window for better display
cv2.namedWindow('VisionTrax - Vehicle Detection & Count', cv2.WINDOW_NORMAL)
cv2.resizeWindow('VisionTrax - Vehicle Detection & Count', 960, 540)  # Reduced window size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce load (process every 3rd frame)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
        start_time = time.time()  # Start timing

        orig_h, orig_w = frame.shape[:2]
        # Further reduced resolution for faster processing
        img = letterbox(frame, new_shape=(224, 224), auto=False)[0]  # Reduced to 224x224
        img_resized = img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5)[0]  # Lowered conf_thres to 0.3

        detections_list = []
        vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape).round()

            for *xyxy, conf, cls in pred:
                cls = int(cls.item())
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        detections_list.append(([x1, y1, w, h], conf.item(), cls))

        if detections_list:
            tracks = deep_sort.update_tracks(detections_list, frame=frame)
        else:
            tracks = []

        # Dynamic virtual line at half the frame height, adjustable with 'u' and 'd'
        line_y = frame.shape[0] // 2
        if cv2.waitKey(1) & 0xFF == ord('u'):
            line_y = max(0, line_y - 10)
        elif cv2.waitKey(1) & 0xFF == ord('d'):
            line_y = min(frame.shape[0], line_y + 10)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            # Assign class if not already set
            if not hasattr(track, 'det_class') and hasattr(track, 'detection_index') and track.detection_index is not None:
                track.det_class = detections_list[track.detection_index][2]

            det_class = getattr(track, 'det_class', None)
            label = f'{CLASS_NAMES.get(det_class, "Vehicle")} ID {track_id}' if det_class is not None else f'ID {track_id}'
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 1)  # Reduced thickness to 1
            cv2.putText(frame, label, (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Smaller text

            if cy > line_y and track_id not in already_counted_ids:
                already_counted_ids.add(track_id)
                if det_class is not None:
                    vehicle_counts[det_class] += 1
                print(f"Counted {CLASS_NAMES.get(det_class, 'Vehicle')} ID {track_id} at y={cy}")

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        y_offset = 30
        for cls_id, count in sorted(vehicle_counts.items()):
            class_name = CLASS_NAMES.get(cls_id, 'Vehicle')
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)  # Smaller text
            y_offset += 20

        cv2.imshow('VisionTrax - Vehicle Detection & Count', frame)
        print(f"Frame processing time: {time.time() - start_time:.2f} seconds")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()