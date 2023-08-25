import cv2
import numpy as np
import time
from mtcnn import MTCNN
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

video_capture = cv2.VideoCapture(r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.webm')

desired_fps = 60
video_capture.set(cv2.CAP_PROP_FPS, desired_fps)

total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)


m = r'C:\Users\Raum\Desktop\jec\code\dataface\dogmeme.png'

frame_number = 0
output_directory = r'C:/Users/Raum/Desktop/jec/code/videos/'

detector = MTCNN()

start = time.time()
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        print("detection found facial person")
        for person in result:
            pass
    else:
        print("no person")

    # cv2.imshow('frame', frame)
    frame_number += 1
    print(frame_number, ": " , total_frames)
    if  cv2.waitKey(1) & 0xff == 27:  # Press 'ESC' for exiting video
        break


video_capture.release()
cv2.destroyAllWindows()

end = time.time()
print(end - start)
