import numpy as np
import cv2

#np.set_printoptions(threshold=np.inf)
cap = cv2.VideoCapture(0)
cap.set(3,480) #horizontal res
cap.set(4,800) #verical res

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame from camera.")
        break
     
    frame = cv2.cvtColor(frame, 0)
    
    detections = cascade_classifier.detectMultiScale(frame)
    
    if(len(detections)>0):
        (x,y,w,h) = detections[0]
        #rect = cv2.rectangle(frame,(x,y),(x+w,y+h),4)
        #print(h)
        im = frame[y:y+h, x:x+h]
        imS = cv2.resize(im, (480, 800), interpolation = cv2.INTER_LINEAR)
        frame = imS
        #cv2.putText(frame,'PERFECT', (40,100), font, 3, (255,255,255), 1, cv2.LINE_AA)
    cv2.namedWindow('OUTPUT', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('OUTPUT', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('OUTPUT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()
    