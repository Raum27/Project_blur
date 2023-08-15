from process_predict import Process_prediction
import cv2
from PIL import Image
import numpy as np
import time

# video_capture = cv2.VideoCapture(r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.webm')
video_capture = cv2.VideoCapture(r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.webm')


total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0

desired_fps = 60  # For example, set to 30 FPS

video_capture.set(cv2.CAP_PROP_FPS, desired_fps)

speed_factor = 2.0

start = time.time()
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # BGR2RGB For Pillow
    small_frame = cv2.resize(frame, (256, 256))
    file_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    # cv2.imshow('frame', frame)
    Process_prediction(file_img)
    if  cv2.waitKey(1) & 0xff == 27:  # Press 'ESC' for exiting video
        break

    
    frame_number += 1
    print(frame_number, ": " , total_frames)

video_capture.release()
cv2.destroyAllWindows()
end = time.time()

print(end - start)
# print("total_frames: ",total_frames)
