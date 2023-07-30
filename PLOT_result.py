from PIL import Image,ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def plot_result(file_image,file_filter,lock_face,embedding_faces,position,check=True):
    '''
    plot_result ใช้เพื่อแสดงผลลัพธ์จากการจับใบหน้า

    file_image: 'path/file/image'
        - ไฟล์ภาพ
        
    file_filter: 'path/file/filter'
        - ไฟล์ภาพ

    lock_face: array 
        - array of image_lock_face

    embedding_faces: feature
        - feature of image_lock_face

    position: array
        - ตำแหน่งของใบหน้าที่จะล็อค

    check : bool default=True
        - ถ้าเป็นTrue หมายความว่าภาพนั้นมาจากไฟล์แต่ถ้าเป็น false มาจาก cv2 ซึ่งเป็น array
    
    return : ภาพที่มีการfilter เรียบร้อยและ ค่าความคล้ายกัน
    
    '''
    log_cosine_similarity = []
    if check:
        image = Image.open(file_image).convert('RGB')
    else:
        image = file_image

    fitter_image = Image.open(file_filter).convert('RGB')
    draw = ImageDraw.Draw(image)

    Detection_check = np.zeros((len(position)))
    for j in range(len(lock_face)):
        for i in range(len(position)):
            similar = cosine_similarity(lock_face[j], embedding_faces[i])
            log_cosine_similarity.append(similar)
            if similar>0.59 and Detection_check[i] ==0 :
                Detection_check[i] = 1
                break


    for i in range(len(position)):
        if Detection_check[i] == 1:
            draw.rectangle([(position[i][0],position[i][1]),(position[i][2],position[i][3])],outline=(25,255,0),width=5)
        else :
            fitter_face = fitter_image.resize((position[i][2]-position[i][0],position[i][3]-position[i][1]))
            image.paste(fitter_face,(position[i][0], position[i][1]))
    return image,log_cosine_similarity
        