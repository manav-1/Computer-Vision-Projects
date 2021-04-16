import numpy as np
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('E:/Manav_Docs/Machine_Learning/LOCKDOWN_PROJECTS/Face_Detection2/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={"person_name":1}
with open('labels.pkl', 'rb' ) as f:
    og_labels=pickle.load(f)
    labels= {v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(1)

while True:
    #capturing Frame by Frame
    ret, frame= cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font= cv2.FONT_HERSHEY_SIMPLEX
            color=(255,100,100)
            stroke=2
            name=labels[id_]
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item="myimg.png"
        cv2.imwrite(img_item, roi_color)#writing the image
        
        color= (255,100,100)
        stroke=2
        end_cord_x= x+w
        end_cord_y= y+h
        font= cv2.FONT_HERSHEY_SIMPLEX
        text=" "+str(x)+","+str(y)+","+str(x+w)+","+str(y+h)+" "
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)        
        
        
    #Display the resulting Frame
    cv2.imshow('WEBCAM',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
#when Everything is done release the Capture
cap.release()
cv2.destroyAllWindows()
