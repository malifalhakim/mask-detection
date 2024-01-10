import os
import random
import time
import cv2
from ultralytics import YOLO

from tracker import Tracker

video_name = f"video-1.mp4"
video_path = os.path.join('.', 'video', video_name)
video_out_path = os.path.join('.', 'video-out',video_name)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model_path = os.path.join('.','runs','detect','train4','weights','best.pt')
model = YOLO(model_path)

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
total_inference_time = 0.0 
while ret:
    frame = cv2.flip(frame, 0)

    
    start_time = time.time()  # Simpan waktu awal inference
    results = model(frame,conf=0.5)
    end_time = time.time()  # Simpan waktu akhir inference
    inference_time = end_time - start_time  # Hitung waktu inference
    total_inference_time += inference_time  # Akumulasikan waktu inference

    '''
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

            start_time = time.time()  # Simpan waktu awal inference
            tracker.update(frame, detections)
            end_time = time.time()  # Simpan waktu akhir inference
            inference_time = end_time - start_time  # Hitung waktu inference
            total_inference_time += inference_time  # Akumulasikan waktu inference

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                text = f"Id {track_id} - mask"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.rectangle(frame, (int(x1) - 1, int(y1) - 20),(int(x1) + len(text) * 12, int(y1)), (colors[track_id % len(colors)]), -1)
                cv2.putText(frame, text, (int(x1) + 5, int(y1) - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    '''
        
        # Without DeepSORT
    frame = results[0].plot()
        
    frame_resized = cv2.resize(frame, (600, 700))
    cv2.imshow('Inference',frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

print(total_inference_time)

cap.release()
cap_out.release()
cv2.destroyAllWindows()
