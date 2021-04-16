import cv2
import os
import numpy as np 
import facedetection as fd

test_img=cv2.imread('E://Manav_Docs//Machine_Learning//LOCKDOWN_PROJECTS//Face_detection//tester//IMG_70952.jpg')
faces_detected,gray_img=fd.faceDetection(test_img)
print("faces detected ",faces_detected)

# for(x,y,w,h) in faces_detected:
# 	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

# resized_img=cv2.resize(test_img,(1000,700))
# cv2.imshow("face detection tutorial", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

faces,face_ID=fd.labels_for_training_data("E://Manav_Docs//Machine_Learning//LOCKDOWN_PROJECTS//Face_detection//trainingimages")
face_recognizer=fd.train_classifier(faces,face_ID)
face_recognizer.save("training_data.yml")
# face_recognizer= cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read("E://Manav_Docs//Machine_Learning//LOCKDOWN_PROJECTS//Face_detection///training_data.yml")
name={0:"Bhumika",1:"Manav"}

for face in faces_detected:
	(x,y,w,h)=face
	roi_gray=gray_img[y:y+h,x:x+w]
	label,confidence=face_recognizer.predict(roi_gray)
	print("confidence:", confidence)
	print("label: ", label)
	fd.draw_rect(test_img,face)
	predicted_name=name[label]
	fd.put_text(test_img,predicted_name,x,y)


resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows