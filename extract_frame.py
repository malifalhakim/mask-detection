import cv2

vidcap = cv2.VideoCapture('Validation Video\Video2\VID_20230622_091545.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("Validation Video\\Video2\\Frame\\frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1