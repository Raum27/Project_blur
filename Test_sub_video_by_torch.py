from process_predict import Process_prediction
import cv2
from PIL import Image
import numpy as np
import time

# video_capture = cv2.VideoCapture(r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.webm')

video_capture = cv2.VideoCapture(r'C:\Users\Raum\Desktop\jec\code\dataface\animal.mp4')
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
output_directory = r'C:/Users/Raum/Desktop/jec/code/videos/'


desired_fps = 35  # For example, set to 30 FPS

# Set the frame width and height (optional)
# frame_width = 640
# frame_height = 480
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Set the frame rate
video_capture.set(cv2.CAP_PROP_FPS, desired_fps)

speed_factor = 2.0

start = time.time()
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR2RGB For Pillow
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('frame', frame)

    if  cv2.waitKey(1) & 0xff == 27:  # Press 'ESC' for exiting video
        break


    output_filename = f"{output_directory}frame_{frame_number}.jpg"
    cv2.imwrite(output_filename,frame)
    frame_number += 1
    print(frame_number, ": " , total_frames)

video_capture.release()
cv2.destroyAllWindows()
end = time.time()

print(end - start)
print("total_frames: ",total_frames)
