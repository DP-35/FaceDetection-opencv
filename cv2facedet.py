# Using pre-trained Haar classifier:::---

import cv2

def draw_bound(img,classifier,scale,minNeigh,color,text):
	imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	features=classifier.detectMultiScale(imgray,scale,minNeigh)
	#scalefactor-scales every large and small faces to a extent
	#minNeighbours-how many neghbours to check inorder to declare whether its a face or not
	face_co=[]
	for (x,y,w,h) in features:
		cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
		cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
		face_co=[x,y,w,h]

	return face_co	


def detect(img,faceclassifier,eyeclassifier):
	color = {"blue":(255,0,0),"red":(0,0,255),"green":(0,255.0)}

	face_co=draw_bound(img,faceclassifier,1.1,10,color['blue'],"Face")
	
	if len(face_co)==4:
		eye_img=img[face_co[1]:face_co[1]+face_co[3],face_co[0]:face_co[0]+face_co[2]]#extrating only face from the whole image
		face_co=draw_bound(img,eyeclassifier,1.1,15,color['red'],"Eye")
	return img

faceclassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeclassifier=cv2.CascadeClassifier("haarcascade_eye.xml")

cap=cv2.VideoCapture(0)

while True: 
	_,img=cap.read()
	img=detect(img,faceclassifier,eyeclassifier)
	cv2.imshow("My face",img)
	if cv2.waitKey(1) & 0xFF==ord('d'):
		break

cap.release()
cv2.destroyAllWindows()


