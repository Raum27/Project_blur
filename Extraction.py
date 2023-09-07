from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

embedder = FaceNet()
detector = MTCNN()

def visual_multi(path_file):

  nplots = sorted(os.listdir(path_file))
  fig = plt.figure(figsize=(8,4))
  for j in range(len(nplots)):
    # print(path_file+nplots[j])
    img = cv2.resize(cv2.imread(path_file+nplots[j]),(224,224))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(4,10,j+1)
    plt.imshow(image_rgb)
    plt.title(j)
    plt.xticks([]);plt.yticks([])
  plt.show()


def extraction_keras_all_images(file_name,show=None): # take embeddings, and position
  position_crop = []
  embedding_faces = []
  position = []
  image = Image.open(file_name).convert('RGB')
  im_arr = np.array(image)
  faces = detector.detect_faces(im_arr,)
  '''for delete file in image_lock'''
  if os.listdir(r'C:\Users\Raum\Desktop\jec\code\image_lock\\') !=[]:
    for i in os.listdir(r'C:\Users\Raum\Desktop\jec\code\image_lock\\'):
      file_image = os.path.join(r'C:\Users\Raum\Desktop\jec\code\image_lock\\',i)
      os.remove(file_image)

  for i in faces:
    x1,y1,width,height = i['box']
    x2,y2 = (x1+width),(y1+height)
    position_crop.append([y1,y2,x1,x2])
    position.append([x1,y1,x2,y2])
    # print(i['box'])

  index_image = 0
  for i in position_crop:
    face = im_arr[i[0]:i[1],i[2]:i[3]]
    image_face = Image.fromarray(face)
    image_face = image_face.resize((224,224))
    faces_crops = np.array(image_face).reshape(-1,224,224,3)
    embedding_faces.append(embedder.embeddings(faces_crops))
    index_image+=1
    path = r'C:\Users\Raum\Desktop\jec\code\image_lock\lock_face_{}.jpg'.format(index_image) # path for collect images
    image_face.save(path)
    
    if show:
      image_face.show()
  visual_multi(r'C:\Users\Raum\Desktop\jec\code\image_lock\\')
  return embedding_faces,position

def extraction_keras_all_video(im_arr):
    position_crop = []
    embedding_faces = []
    position = []
    faces = detector.detect_faces(im_arr)
    
    if faces == []:
        return [],[]

    for i in faces:
        x1,y1,width,height = i['box']
        x2,y2 = (x1+width),(y1+height)
        position_crop.append([y1,y2,x1,x2])
        position.append([x1,y1,x2,y2])

    # print(len(position_crop))
    for i in position_crop:
        # print(i)
        face = im_arr[i[0]:i[1],i[2]:i[3]]
        image_face = cv2.resize(face,(224,224))
        faces_crops = np.expand_dims(image_face,axis=0)
        embedding_faces.append(embedder.embeddings(faces_crops))
        # print(embedding_faces)
    return embedding_faces,position

def extraction_keras_lock(*file_name,show=None):
  if type(file_name) is tuple:
    file_name = [*file_name,]
  result = []
  for i in file_name:
    image = Image.open(i).convert('RGB')
    im_arr = np.array(image)
    N=0
    faces = detector.detect_faces(im_arr)
    x1,y1,width,height = faces[N]['box'] # faces[N]['box']
    x2,y2 = (x1+width),(y1+height)
    face = im_arr[y1:y2,x1:x2] # ตัดเอาเฉพาะใบหน้า
    image_face = Image.fromarray(face)
    image_face= image_face.resize((224,224))
    face_img = np.expand_dims(image_face,axis=0)
    result.append(embedder.embeddings(face_img))

  return result