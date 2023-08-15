from torchvision.io import read_image
from torchvision.models import  MobileNet_V3_Large_Weights
from model import Model_pretrain
import cv2
import torch
from PIL import Image
import time
import os
def Process_prediction(file_img):
    model ,weights = Model_pretrain()
    img = Image.fromarray(cv2.cvtColor(cv2.resize(cv2.imread(file_img),(512,512)), cv2.COLOR_BGR2RGB))

    # img = Image.fromarray(file_img) # case for process already
    # img = read_image(file_img)

    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0).to('cuda')
    # print(batch.size())
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    # Process_prediction(r'C:\Users\Raum\Desktop\jec\code\videos\frame_32784.jpg')
    start = time.time()
    dir_file = os.listdir(r'C:\Users\Raum\Desktop\jec\code\videos\\')
    print(len(dir_file))  
    for i in range(2000):
         Process_prediction(f'C:/Users/Raum/Desktop/jec/code/videos/{dir_file[i]}')  
    end = time.time()
    print(end - start)