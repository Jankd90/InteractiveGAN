import cv2

def draw_circle(event,x,y,flags,param):
	global img,pointIndex,pts

	if event == cv2.EVENT_LBUTTONDOWN:
		if(pointIndex<4):
			print(pointIndex)
			cv2.circle(img,(x,y),10,colors[pointIndex],-1)
			pts[pointIndex] = (x,y)
			pointIndex = pointIndex + 1