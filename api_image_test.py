from Extraction import extraction_keras_all_images
from Annotation import annotation_box_images

f = r'C:\Users\Raum\Desktop\jec\code\dataface\avenger.jpg'
m = r'C:\Users\Raum\Desktop\jec\code\dataface\dogmeme.png'
embedding_faces,position=extraction_keras_all_images(f) # lock by self-image

img,_ =annotation_box_images(f,embedding_faces,position,lock_face=[0,8,4],file_filter=None)
img.show()
