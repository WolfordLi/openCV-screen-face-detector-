from PIL import ImageGrab
import cv2
import numpy as np
face = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
while True :
    capture = ImageGrab.grab()
    image = cv2.cvtColor(np.array(capture), cv2.COLOR_RGB2BGR)
    gray_verion = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    face_result = face.detectMultiScale(gray_verion)
    for x,y,w,h in face_result:
        cv2.circle(image, center = (x+w//2, y+h//2),radius=w//2,color=[0,0,255],thickness=7)
        position = (x+w//2, y+h//2)
        print("head:",position)   
    cv2.namedWindow('screen detection', 0)
    cv2.resizeWindow('screen detection',480, 270)
    cv2.imshow('screen detection', image)
    key = cv2.waitKey(1000//24)
    if key == 27:
        break
cv2.destroyAllWindows()
capture.release()
exit()



