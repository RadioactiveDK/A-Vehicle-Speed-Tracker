import jetson.inference
import jetson.utils
import time
import cv2
import math
import numpy as np 

# initializing some varables
timeStamp=time.time()
fpsFilt=0
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
dispW=1280
dispH=720
font=cv2.FONT_HERSHEY_SIMPLEX
angle=50*3.1415/180
ptime = time.time()

# taking input for vehicle distance
dist=float(input("Enter approx distance from road in m: "))

# activating camera
cam=cv2.VideoCapture('/dev/video0')

cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
ppos=(0.0,0.0)
fps=0

while True:
    _,img = cam.read()
    height=img.shape[0]
    width=img.shape[1]

    # converting to appropriate format so that we can use object detection
    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    frame=jetson.utils.cudaFromNumpy(frame)

    # performing object detection
    detections=net.Detect(frame, width, height)
    for detect in detections:
        # print(detect)
        ID=detect.ClassID
        top=int(detect.Top)
        left=int(detect.Left)
        bottom=int(detect.Bottom)
        right=int(detect.Right)
        item=net.GetClassDesc(ID)

        # selecting the object of interest
        if item=='car':
            # finding speed and distance moved
            horizDist=( (left+right)/2 -ppos[0] ) * 1/1280 * dist * 2 * math.tan(angle/2) 
            if horizDist<0:
                horizDist = horizDist*(-1)
            speed = horizDist*fpsFilt
            print(str(round(speed,2))+'m/s')
            ppos=((left+right)/2,(top+bottom)/2)

            # marking object on the frame
            cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
            cv2.putText(img,str(round(speed,2))+'m/s',(left,top+20),font,1,(0,0,255),2)
    # updating fps
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps

    # displaying fps
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)

    # displaying the processed frame
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)

    # to exit the code, press 'q'
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
