import cv2
import numpy as np
import time
import threading

with open(r"C:\Users\Raum\Desktop\array_of_frame\2023-09-07_14-56-57_Video_array.npy", 'rb') as f:
    a = np.load(f)
print(a.shape)

start = time.time()
image_files = []
path_file ="C:/Users/Raum/Desktop/array_of_frame/allframe/"

for i in range(a.shape[0]):
    # frame = cv2.imread(path_file+str(i)+'.jpg')
    image_files.append(path_file+str(i)+'.jpg')
    
def read_image(file_path,image_all = []):
    image = cv2.imread(file_path)
    print(f"Image {file_path} dimensions: {image.shape}")
    image_all.append(image)

threads = []
image_all = []
for file_path in image_files:
    thread = threading.Thread(target=read_image, args=(file_path,image_all))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# print(threads)
print(len(threads))
print(image_all[0].shape)
print("All threads have finished.")

end = time.time()
print(end - start)