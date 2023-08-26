import cv2

# Open the video file
video_path = r'C:\Users\Raum\Desktop\jec\code\dataface\videoplayback.mp4'
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Write the resized frame to the output video
    out.write(resized_frame)
    
    # cv2.imshow('Resized Video', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
