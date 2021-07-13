import cv2,os
import numpy as np

from keras.utils import np_utils
from keras.models import load_model

net = cv2.dnn.readNet("darknet/cfg/yolov3.cfg", "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('FaceClassifier.hdf5')


labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}

source=cv2.VideoCapture(-1)
# source=cv2.VideoCapture("s5.mp4")

# frame_width = int(source.get(3)) 
# frame_height = int(source.get(4)) 
   
# size = (frame_width, frame_height)
# result_cap = cv2.VideoWriter('fs5.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          30, size)

# %% [code]
while(True):
	
	ret,img=source.read()
	out_img=img
	(H, W) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

	gray=cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
	if len(idxs) > 0:
		for i in idxs.flatten():
			if(classIDs[i] != 0):
				continue
			(x1, y1) = (boxes[i][0], boxes[i][1])
			(w1, h1) = (boxes[i][2], boxes[i][3])
			cv2.rectangle(out_img, (x1, y1), (x1 + w1, y1 + h1), (255,0,0), 2)
			person_img=gray[y1:y1+h1,x1:x1+w1]
			faces=face_clsfr.detectMultiScale(person_img,scaleFactor=1.3,minNeighbors=5)  

			for (x,y,w,h) in faces:
				# print('face detected')
				face_img=person_img[y:y+w,x:x+w]
				resized=cv2.resize(face_img,(100,100))
				normalized=resized/255.0
				reshaped=np.reshape(normalized,(1,100,100,1))
				result=model.predict(reshaped)

				label=np.argmax(result,axis=1)[0]

				cv2.rectangle(out_img,(x1+x,y1+y),(x1+x+w,y1+y+h),color_dict[label],2)
				cv2.rectangle(out_img,(x1+x,y1+y-40),(x1+x+w,y1+y),color_dict[label],-1)
				cv2.putText(out_img, labels_dict[label], (x1+x, y1+y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

	cv2.imshow('LIVE',img)
	# result_cap.write(out_img)
	key=cv2.waitKey(1)

	if(key!=-1):
		break
        
source.release()
cv2.destroyAllWindows()