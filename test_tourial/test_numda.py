from numba import njit,jit
import tensorflow as tf
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# ไม่ใช้ Numba
def f1(x, y):
  while x<y:
    x+=1
  return x

# ใช้ Numba
@jit(nopython=True)
def f2(x, y):
  while x<y:
    x+=1
  return x


# ใช้ Numba
@jit(nopython=False, parallel=True,forceobj=True)
def f3():

  video_capture = cv2.VideoCapture(r"C:\Users\Raum\Desktop\jec\code\dataface\walk.mp4")
  output_directory="C:/Users/Raum/Desktop/array_of_frame/allframe/"
  TOTAL_FRAMES = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
  print(TOTAL_FRAMES)
  MASK_FRAMES = np.zeros(TOTAL_FRAMES,dtype=np.int32)
  
  pos_frame = 0
  file_path = 'C:/Users/Raum/Desktop/array_of_frame/allframe'

  ''' pull image from folder '''
  frame_maker = cv2.resize(cv2.imread(f'C:/Users/Raum/Desktop/array_of_frame/allframe/frame_{str(0)}.jpg'),(512,512)).reshape(-1)
  print(frame_maker.shape)
  start = time.time()
  for i in range(TOTAL_FRAMES):
    frame = cv2.resize(cv2.imread(f'C:/Users/Raum/Desktop/array_of_frame/allframe/frame_{str(i)}.jpg'),(512,512)).reshape(-1)
    if cosine_similarity([frame_maker],[frame])[0,0]>= 0.8:
        MASK_FRAMES[i]=pos_frame
    elif cosine_similarity([frame_maker],[frame])[0,0]== 0.0:
        MASK_FRAMES[i]=pos_frame
    else :
        frame_maker = cv2.resize(cv2.imread(f'C:/Users/Raum/Desktop/array_of_frame/allframe/frame_{str(i)}.jpg'),(512,512)).reshape(-1)
        pos_frame +=1
        MASK_FRAMES[i]=pos_frame

  end = time.time()
  print(end-start)
  return MASK_FRAMES


if __name__ == "__main__":
    # strat = time.time()
    # print(f1(0,1000000000))
    # end = time.time()
    # print(end-strat)

    # strat = time.time()
    # print(f2(0,1000000000))
    # end = time.time()
    # print(end-strat)
    a = f3()
    print(a)