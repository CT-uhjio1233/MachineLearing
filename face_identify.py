import cv2 as cv

face_cade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img=cv.imread('/Users/ct/Documents/machine_learing/test_data/td1.jpeg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces=face_cade.detectMultiScale(
    gray,
    scaleFactor=1.08,
    minNeighbors=5,
    minSize=(32,32)
    )

for(x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.namedWindow('img',cv.WINDOW_NORMAL)
cv.imshow('img',img)
cv.waitKey
cv.destroyAllWindows()