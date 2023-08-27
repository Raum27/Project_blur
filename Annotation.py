from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image,ImageDraw
import cv2

import numpy as np


def annotation_box_pillow(file_image,lock_face,embedding_faces,position,file_filter=None):
  log_cosine_similarity = []
  image = Image.open(file_image).convert('RGB')
  if file_filter is None:
      draw = ImageDraw.Draw(image)
  else:
    fitter_image = Image.open(file_filter).convert('RGBA')


  Detection_check = np.zeros((len(position)))
  for j in range(len(lock_face)):
    for i in range(len(position)):
          similar = cosine_similarity(embedding_faces[lock_face[j]], embedding_faces[i])
          log_cosine_similarity.append(similar)
          if similar>0.9 and Detection_check[i] ==0 :
            Detection_check[i] = 1
            break


  for i in range(len(position)):
      if Detection_check[i] == 1:
        # draw.rectangle([(position[i][0],position[i][1]),(position[i][2],position[i][3])],outline=(25,255,0),width=5)
        pass
      else :
        x = int((position[i][2]-position[i][0]))
        y = int((position[i][3]-position[i][1]))
        # print(x,y)
        if file_filter is None:
          censor_region = (position[i][0],position[i][1], position[i][2],position[i][3])  # Format: (left, top, right, bottom)
          censored_area = image.crop(censor_region)


          censored_width, censored_height = censored_area.size
          pixel_size = 10 
          censored_area = censored_area.resize(
              (censored_width // pixel_size, censored_height // pixel_size),
              Image.NEAREST
          ).resize(
              (censored_width, censored_height),
              Image.NEAREST
          )
          image.paste(censored_area, censor_region)

        else:
          fitter_face = fitter_image.resize((x,y))
          image.paste(fitter_face,(position[i][0], position[i][1]),fitter_face) # gan:X, gan:Y
      
        
  return image,log_cosine_similarity

def annotation_box_cv2(file_image,lock_face,embedding_faces,position,file_filter=None):
    log_cosine_similarity = []
    Detection_check = np.zeros((len(position)))
    for j in range(len(lock_face)):

        for i in range(len(position)):
            similar = cosine_similarity(lock_face[j], embedding_faces[i])
            log_cosine_similarity.append(similar)
            if similar>0.55 and Detection_check[i] ==0 :
                Detection_check[i] = 1
                break

    for i in range(len(position)):
        if Detection_check[i] == 1:
            pass
        else :
            censor_region = (position[i][0],position[i][1],position[i][2],position[i][3])

            censored_area = file_image[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]]

            censored_width, censored_height = censored_area.shape[1], censored_area.shape[0]


            pixel_size = 12
            censored_area = cv2.resize(censored_area, (censored_width // pixel_size, censored_height // pixel_size))
            censored_area = cv2.resize(censored_area, (censored_width, censored_height), interpolation=cv2.INTER_NEAREST)

            file_image[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]] = censored_area
    
    return file_image