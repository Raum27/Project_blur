import cv2
import time

# Open the video file
video_path = r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.mp4'
cap = cv2.VideoCapture(video_path)


output_directory=r"C:\Users\Raum\Desktop\array_of_frame\allframe\\"
frame_number=0

start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(output_directory+str(frame_number)+'.jpg',frame)
    frame_number+=1

cap.release()

end = time.time()
print(end - start)