# # import torch
# import cv2
# import os
# import time
# from mtcnn.mtcnn import MTCNN
# # from keras_facenet import FaceNet

# # model =torch.hub.load(".",'custom',path="C:/Users/Raum/Desktop/yolo/yolov5/crowdhuman_yolov5m.pt",source='local')
# detector = MTCNN()
# # print(os.listdir("C:/Users/Raum/Desktop/array_of_frame/allframe/"))
# start = time.time()
# for i in range(len(os.listdir("C:/Users/Raum/Desktop/array_of_frame/allframe/"))):
#     im = f"C:/Users/Raum/Desktop/array_of_frame/allframe/frame_{i}.jpg"
#     im = cv2.imread(im)
#     img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(img,)
#     # results = model(img)
#     # print(i,"  ",img.shape)
# end = time.time()
# print(end-start)
# # print(results.pand


import concurrent.futures
import time

def thread_function(thread_id):
    print(f"Thread {thread_id} started")
    # time.sleep(2)
    print(f"Thread {thread_id} finished")

def main():
    # Set the maximum number of threads
    max_threads = 3

    # Create a ThreadPoolExecutor with the specified maximum number of threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        # Submit tasks to the thread pool
        thread_ids = range(1, 6)
        futures = {executor.submit(thread_function, thread_id): thread_id for thread_id in thread_ids}

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
