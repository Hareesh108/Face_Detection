import  cv2
#create cascade classifier object
face_cascade = cv2.CascadeClassifier("D:\Projects\Face detection\haarcascade_frontalface_default.xml")

#read image as it is
img = cv2.imread("D:\Projects\Face detection\Image.jpg",1)

#reading the image as gray scale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#search the image as gray scale image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05 , minNeighbors=50)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)

#resize the images
# resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

cv2.imshow("Legend",img)
cv2.waitKey(0) 

cv2.destroyAllWindows
# print(img.shape)
