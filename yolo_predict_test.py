import os
import re
import natsort 
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random

model_path = os.path.join('.','runs','detect','train4','weights','best.pt')
model = YOLO(model_path)

file_directory = os.path.join('.','DatasetV2','test','video-1')
list_file = os.listdir(file_directory)
list_file = natsort.natsorted(list_file)

label_directory = os.path.join('.','DatasetV2','test','label-1')
list_label = os.listdir(label_directory)
list_label = natsort.natsorted(list_label)

save_dir = os.path.join('.','DatasetV2','test','pred-1')

video_out_path = os.path.join('.','video-out','video-1')

print("START!")

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255),255) for j in range(10)]

detection_threshold = 0.5
for i in range(len(list_file)):
    filename = list_file[i]
    labelname = list_label[i]

    label_path = os.path.join(save_dir,list_label[i])
    file_path = os.path.join(file_directory,filename)

    with open(label_path,'w+') as f:

        frame = cv2.imread(file_path)
        
        # Run YOLOv8 inference on the frame
        results = model(frame)

        for result in results:
            scores = []
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                scores.append(score)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            if detections:
                tracker.update(frame, detections)

                idx = 0
                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox

                    x1 = 0 if x1 < 0 else x1
                    y1 = 0 if y1 < 0 else y1
                    x2 = 0 if x2 < 0 else x2
                    y2 = 0 if y2 < 0 else y2
                    
                    track_id = track.track_id
                    try: 
                        writted_str = f"{class_id} {scores[idx]} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                        f.write(writted_str)
                        text = f"Id {track_id} - mask"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                        cv2.rectangle(frame, (int(x1) - 1, int(y1) - 20),(int(x1) + len(text) * 12, int(y1)), (colors[track_id % len(colors)]), -1)
                        cv2.putText(frame, text, (int(x1) + 5, int(y1) - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except:
                        print("Masalah prediksi ke",idx)
                    idx += 1 

        # Without DeepSORT
        frame = cv2.resize(frame,(400,600))
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
        cv2.imwrite(video_out_path+filename,frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()