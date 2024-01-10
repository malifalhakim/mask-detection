import cv2
import os
import natsort

file_directory = os.path.join('.','dataset','test','video-1')
list_file = os.listdir(file_directory)
list_file = natsort.natsorted(list_file)

label_directory = os.path.join('.','dataset','test','label-1')
list_label = os.listdir(label_directory)
list_label = natsort.natsorted(list_label)

save_dir = os.path.join('.','dataset','test','ground-truth-1')

def read_label(filename,labelname,save_path):
    txtname = labelname
    save_dir = os.path.join(save_path,txtname)

    image_path = os.path.join(file_directory,filename)
    label_path = os.path.join(label_directory,labelname)

    img = cv2.imread(image_path)
    with open(label_path,'r') as f:
        with open(save_dir,'w+') as f2:
            annotations = f.readlines()
            for annotation in annotations:
                tokens = annotation.split(" ")
                if len(tokens) == 5:
                    class_name = tokens[0]
                    x_center = float(tokens[1])
                    y_center = float(tokens[2])
                    width = float(tokens[3])
                    height = float(tokens[4])

                    x_min = (x_center - width/2) * img.shape[1]
                    x_max = (x_center + width/2) * img.shape[1]
                    y_min = (y_center - height/2) * img.shape[0]
                    y_max = (y_center + height/2) * img.shape[0]

                    if x_min < 0 or x_max < 0  or y_min <0 or y_max < 0:
                        print(f"Filename {filename} Ada Negatif")

                    x_min = 0 if x_min < 0 else x_min
                    x_max = 0 if x_max < 0 else x_max
                    y_min = 0 if y_min < 0 else y_min
                    y_max = 0 if y_max < 0 else y_max

                    writted_str = f"{class_name} {x_min} {y_min} {x_max} {y_max}\n"
                    f2.write(writted_str)

for i in range(len(list_file)):
    read_label(list_file[i],list_label[i],save_dir)
    
    