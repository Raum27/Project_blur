import cv2
import time
import os
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*'XVID') # *'mp4v',  *'mov''''

sstrr = time.strftime("%Y-%m-%d_%H-%M-%S")

output_path = f'C:/Users/Raum/Desktop/writed_video/{sstrr}_Video.avi'

new_width = 1920
new_height =1080

out = cv2.VideoWriter(output_path, fourcc, 20, (new_width, new_height))

with open(r"C:\Users\Raum\Desktop\array_of_frame\2023-09-07_14-56-57_Video_array.npy", 'rb') as f:
    a = np.load(f)
print(a.shape)

start = time.time()
image_files = []
path_file ="C:/Users/Raum/Desktop/array_of_frame/allframe/"

for i in range(a.shape[0]):
    print(path_file+str(i)+'.jpg')
    frame = cv2.imread(path_file+str(i)+'.jpg')
    image_files.append(frame)


# out.release()

cv2.destroyAllWindows()
print(image_files[0].shape)
end = time.time()
print(end - start)