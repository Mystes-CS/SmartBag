import io
from picamera.array import PiRGBArray
from picamera import PiCamera
import sys
sys.path.append('/home/pi/cvpi2/lib/python3.5/site-packages')
sys.path.remove('/usr/local/lib/python3.5/dist-packages')
import cv2
from PIL import Image
import numpy as np
import time
import json
import os
import math


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        
        #PIL_img = Image.open(imagePath).convert('L')
        #PIL_img = cv2.equalizeHist(PIL_img)
        total=0
        for row in PIL_img:
            for value in row:
                 total+=value
        avg = total/10000
        for row in range(len(PIL_img)):
            for col in range(len(PIL_img[row])):
                PIL_img[row][col] = ((PIL_img[row][col]/avg)*135)
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)
        faces.append(img_numpy)
        ids.append(id)
        cv2.imshow("training",img_numpy)
        cv2.waitKey(10)
        #faces = detector.detectMultiScale(img_numpy)
        #for (x,y,w,h) in faces:
         #   faceSamples.append(img_numpy[y:y+h,x:x+w])
         #   ids.append(id)
    return faces,ids

#first = input("Is this your first time using this system? if Yes, input number '1' ; if not, input number '2': ")
face_id = 0
userData = json.loads(open('/home/pi/Downloads/opencv-master/data/face/userInfo.txt').read())
userList = ['']
name = input("Please input your name: ")
for member in userData['User'] :
    face_id+=1
    if member["name"] == name:
        print("This name has been registered! Please input another name, thank you")
        exit()
blu = input("Please input your bluetooth address(Capital): ")
userData['User'].append({"name": name,"BTAddress":blu,"number":str(face_id+1)})
print(userData)
with open('/home/pi/Downloads/opencv-master/data/face/userInfo.txt','w') as file:
    json.dump(userData, file)
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
recognizer = cv2.face.FisherFaceRecognizer_create()
display_window = cv2.namedWindow("Faces")
display_window2 = cv2.namedWindow("Faces2")
time.sleep(0.1)
s=0.5
count = 0
#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-master/data/face/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-master/data/face/haarcascade_eye_tree_eyeglasses.xml')
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    #image = cv2.resize(image,None, fx=s,fy=s,interpolation=cv2.INTER_AREA)
    image2 = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            if eyes[0][0] < eyes[1][0]:
                (x1,y1),(x2,y2) = (eyes[0][0]+eyes[0][2]/2,eyes[0][1]+eyes[0][3]/2),(eyes[1][0]+eyes[1][2]/2,eyes[1][1]+eyes[1][3]/2)
            else:
                (x2,y2),(x1,y1) = (eyes[0][0]+eyes[0][2]/2,eyes[0][1]+eyes[0][3]/2),(eyes[1][0]+eyes[1][2]/2,eyes[1][1]+eyes[1][3]/2)
        
            A = np.zeros((2,3))
            A[0,0:2]=[x2-x1,y2-y1]
            print('start:')
            print(A)
            s = (A[0,0]**2+A[0,1]**2)**0.5
            print(s)
            A[0,0:2]/=s
            A[1,0:2]=[-A[0,1],A[0,0]]
            print(A)
            A[0,:] *= 50/s
            print(A)
            A[1,:] *= 95/(h-(y1+y2)/2)
            print(A)
            A[0,2] = 50-A[0,0]*(x+(x1+x2)/2)-A[0,1]*(y+(y1+y2)/2)
            print(A)
            A[1,2] = 25-A[1,0]*(x+(x1+x2)/2)-A[1,1]*(y+(y1+y2)/2)
            print(A)
            print('end')
            dst_img = cv2.warpAffine(image,A,(100,120))
            temp = cv2.cvtColor(dst_img,cv2.COLOR_BGR2GRAY)
            temp = cv2.resize(temp,(100,100))
            #temp = cv2.equalizeHist(temp)
            count+=1
            cv2.imwrite("dataset/User."+str(face_id+1)+'.'+str(count)+".jpg", temp)
            print("count: "+str(count))
            cv2.imshow("Faces2", temp)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image2,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(0,255,0),2)
    cv2.imshow("Faces", image2)
    key = cv2.waitKey(1)
    rawCapture.truncate(0)

    if key == ord("q"): # Press 'q' to quit
        break
    elif count>29:
        break
cv2.destroyAllWindows()
faces,ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()
