from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.io import read_image
import cv2
import torch
from PIL import Image

file_img = r'C:\Users\Raum\Desktop\jec\code\videos\frame_62094.jpg'
# img = torch.tensor(cv2.resize(cv2.imread(file_img),(512,512)))


# img = Image.open(file_img ).convert('RGB').resize((512,512))
# img = read_image(file_img)
# print(img.shape)



# Step 1: Initialize model with the best available weights
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights).to('cuda')
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0).to('cuda')

print(batch.size())


# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

