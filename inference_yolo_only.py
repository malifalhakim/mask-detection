import os
import re
import natsort 
import cv2
from ultralytics import YOLO
import random

model_path = os.path.join('.','runs','detect','train4','weights','best.pt')
model = YOLO(model_path)

file_directory = os.path.join('.','dataset','test','video-1')
list_file = os.listdir(file_directory)
list_file = natsort.natsorted(list_file)

label_directory = os.path.join('.','dataset','test','label-1')
list_label = os.listdir(label_directory)
list_label = natsort.natsorted(list_label)

save_dir = os.path.join('.','dataset','test','preds-1')
video_out_path = os.path.join('.','video-out','video-1')

print("START!")

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
for i in range(len(list_file)):
    filename = list_file[i]
    labelname = list_label[i]

    label_path = os.path.join(save_dir,labelname)
    file_path = os.path.join(file_directory,filename)

    with open(label_path,'w+') as f:

        frame = cv2.imread(file_path)
        
        #resize frame
        #frame = cv2.resize(frame, (800, 600))
        
        # Run YOLOv8 inference on the frame
        results = model(frame,conf=0.5)

        '''
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    writted_str = f"{class_id} {score} {x1} {y1} {x2} {y2}\n"
                    f.write(writted_str)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[0]), 3)
        '''
                    

        # Without DeepSORT
        frame = results[0].plot()
        frame = cv2.resize(frame,(700,600))
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
        cv2.imwrite(os.path.join(video_out_path,filename),frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()