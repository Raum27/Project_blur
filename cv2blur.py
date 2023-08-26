import cv2

input_image_path = r'C:\Users\Raum\Desktop\jec\code\dataface\superman4.jpg'
output_image_path = "censored_image.jpg"
image = cv2.imread(input_image_path)

censor_region = (100, 200, 300, 400)

censored_area = image[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]]

censored_width, censored_height = censored_area.shape[1], censored_area.shape[0]


pixel_size = 12
censored_area = cv2.resize(censored_area, (censored_width // pixel_size, censored_height // pixel_size))
censored_area = cv2.resize(censored_area, (censored_width, censored_height), interpolation=cv2.INTER_NEAREST)

image[censor_region[1]:censor_region[3], censor_region[0]:censor_region[2]] = censored_area


cv2.imshow('Censored Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()