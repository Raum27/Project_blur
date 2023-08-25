import cv2


image_path = r"C:\Users\Raum\Desktop\jec\code\dataface\superman4.jpg"
image = cv2.resize(cv2.imread(image_path),(720,720))

start_x, start_y, end_x, end_y = 100, 150, 300, 300  # Adjust these values
censored = image.copy()
cv2.rectangle(censored, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)  # Draw a black rectangle

censored = image.copy()
region = censored[start_y:end_y, start_x:end_x]
print(region.shape)
factor = 10  # Adjust the pixelation factor

small = cv2.resize(region, (region.shape[1] // factor, region.shape[0] // factor))
censored[start_y:end_y, start_x:end_x] = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)


cv2.imshow('Censored Image', censored)
cv2.waitKey(0)
cv2.destroyAllWindows()