import cv2,os
import numpy as np

from keras.utils import np_utils
from keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input





net = cv2.dnn.readNet("darknet/cfg/yolov3.cfg", "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

facenet=cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000.caffemodel")
model = load_model('FaceClassifier.hdf5')

labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}

source=cv2.VideoCapture(-1)
# source=cv2.VideoCapture("s6.mp4")

# frame_width = int(source.get(3)) 
# frame_height = int(source.get(4)) 
   
# size = (frame_width, frame_height)
# result_cap = cv2.VideoWriter('s6.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          30, size)

img_size=100
while(True):
	
	ret,img=source.read()# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
			(x2, y2) = (boxes[i][0], boxes[i][1])
			(w2, h2) = (boxes[i][2], boxes[i][3])
			cv2.rectangle(out_img, (x2, y2), (x2 + w2, y2 + h2), (255,0,0), 2)
			person_img=out_img[y2:y2+h2,x2:x2+w2]

			(h, w) = person_img.shape[:2]
			if(len(person_img)==0):
				continue
			blob = cv2.dnn.blobFromImage(cv2.resize(person_img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
			facenet.setInput(blob)
			faces=facenet.forward()

			flag=0
			for i in range(0,faces.shape[2]):
				confidence = faces[0, 0, i, 2]
				if confidence > 0.5:
					box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
					(x, y, x1, y1) = box.astype("int")
					(x,y)=( max(0,x), max(0,y))
					(x1,y1)=( min(w-1,x1), min(h-1,y1))

					face=person_img[y:y1, x:x1]
					face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
					face=cv2.resize(face,(img_size,img_size))
					face=face/255.0
					face=np.reshape(face,(1,img_size,img_size,1))
					(mask, withoutMask) = model.predict(face)[0]
					idx = 0 if mask > withoutMask else 1
					label= labels_dict[idx]
					if(idx==0):
						flag=1
					color = color_dict[idx]
					label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

					cv2.putText(out_img, label, (x2+x,y2+y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(out_img, (x2+x, y2+y),(x2+x1, y2+y1), color, 2)

			if(flag==1):
				cv2.putText(out_img, "DOOR CLOSED", (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			else:
				cv2.putText(out_img, "DOOR OPEN", (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

	cv2.imshow('LIVE',img)
	# result_cap.write(out_img)
	key=cv2.waitKey(1)

	if(key!=-1):
		break
        
source.release()
cv2.destroyAllWindows()