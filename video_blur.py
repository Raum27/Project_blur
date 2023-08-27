import cv2
from Extraction import extraction_keras_all_cv2,extraction_keras_lock
from Annotation import annotation_box_cv2
import time

# Open the video file
video_path = r'C:\Users\Raum\Desktop\jec\code\dataface\netflix.mp4'
cap = cv2.VideoCapture(video_path)

# Define the new width and height for resizing
new_width = 640
new_height = 480

# Get the frames per second (fps) and frame size of the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the resized video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = r'C:\Users\Raum\Desktop\jec\code\experimental\array_of_frame\resized_video.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))


output_directory=r'C:\Users\Raum\Desktop\jec\code\experimental\array_of_frame\\'
frame_number=0
lock_face = extraction_keras_lock(r'C:\Users\Raum\Desktop\jec\code\experimental\lufy.png')
print(lock_face[0].shape)
start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    embedding_faces,position = extraction_keras_all_cv2(resized_frame)
    if embedding_faces==[]:
        print("noononononononononnohuman")
        out.write(resized_frame)
        cv2.imshow('Resized Video', resized_frame)
    else:
        _frame = annotation_box_cv2(resized_frame,lock_face,embedding_faces,position)
        print("found",_frame.shape)
        # output_filename = f"{output_directory}frame_{frame_number}.jpg"
        # cv2.imwrite(output_filename,frame)
        # frame_number += 1
        # Write the resized frame to the output video
        out.write(_frame)
        cv2.imshow('Resized Video', _frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

end = time.time()
print(end - start)