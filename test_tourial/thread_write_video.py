import cv2
import numpy as np
import time
import threading

with open(r"C:\Users\Raum\Desktop\array_of_frame\2023-09-07_14-56-57_Video_array.npy", 'rb') as f:
    a = np.load(f)
print(a.shape)

start = time.time()
image_files = []

sstrr = time.strftime("%Y-%m-%d_%H-%M-%S")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "C:/Users/Raum/Desktop/writed_video/{}_Video_thread.avi".format(sstrr)
new_width = 1920
new_height = 1080
fps = 25

out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

path_file ="C:/Users/Raum/Desktop/array_of_frame/allframe/"
for i in range(a.shape[0]):
    # print(path_file+str(i)+'.jpg')
    image_files.append(path_file+str(i)+'.jpg')


def read_image_and_write(file_path,out):
    for i in file_path:
        print(i)
        frame = cv2.imread(i)
    #     out.write(frame)
    # out.release()

threads = []

for i in range(10):
    thread = threading.Thread(target=read_image_and_write, args=(image_files,out))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("All threads have finished.")

end = time.time()
print(end - start)