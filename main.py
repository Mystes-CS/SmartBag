import io
import RPi.GPIO as GPIO #Import GPIO library
from bluepy.btle import Scanner, DefaultDelegate
from bluetooth import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import bluetooth
import threading
from PIL import Image
import numpy as np
import time
import os
import math
import json
import multiprocessing
from multiprocessing import Process, Lock, freeze_support, set_start_method
import time
#from face_detectForMain import detect
import sys
sys.path.append('/home/pi/cvpi2/lib/python3.5/site-packages')
sys.path.remove('/usr/local/lib/python3.5/dist-packages')
import cv2
flag = 0
List = ['']
userList = ['']
BLUList = ['']
camera = PiCamera()
class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)
        
    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print("Discovered Beacon device", dev.addr, "with RSSI:",dev.rssi," dB")
            List.append(dev.addr)
        elif isNewData:
            print ("Received new data from Beacon device addr:", dev.addr)

def send_leave(q):
    arduino_addr = "98:D3:32:21:2E:85"
    port = 1
    sock = BluetoothSocket(RFCOMM)
    print('Connecting Arduino')
    #sock.connect_ex returns 0 if connected , return errno if connect failed.
    chance = 0
    while sock.connect_ex((arduino_addr,port)) != 0 and chance < 5:
        
        time.sleep(1.0)
        chance = chance + 1
        continue
    if chance == 5:
     print("No Bag Detected!!!")
     buzzerFailed_NoBag()
     q.put(0)
     return
    #Send msg 'leave' and break when received 'ACK'
    str = ''
    while 1:
        print('Sending MSG: leave')
        time.sleep(2.0)
        sock.send('leave')
        time.sleep(2.0)
        data = sock.recv(10)
        #print(data)
        if data == b'ACK\r\n' or str == b'ACK\r\n':
            print('Received ACK')
            break;
        else:
            str = str + data.encode('utf-8')
            continue
    print('Disconnected')
    sock.close()
    q.put(1)
    return

def buzzerAccept():
    GPIO.setup(12,GPIO.OUT)
    p = GPIO.PWM(12,50)
    p.start(15)
    def play(p,frequency,tempo):
        p.ChangeFrequency(frequency)
        time.sleep(0.5*tempo)
        
    play(p,698,0.5)
    play(p,880,0.5)
    play(p,1047,0.5)
    play(p,880,0.5)
    play(p,1047,1)

def buzzerFailed_NoBag():
    GPIO.setup(12,GPIO.OUT)
    p = GPIO.PWM(12,50)
    p.start(15)
    def play(p,frequency,tempo):
        p.ChangeFrequency(frequency)
        time.sleep(0.5*tempo)
        
    play(p,666,1)
    time.sleep(0.5)
    play(p,666,1)
    time.sleep(0.5)
    play(p,666,1)
    time.sleep(0.5)
    play(p,666,1)

def buzzerFailed_NoPhone():
    GPIO.setup(12,GPIO.OUT)
    p = GPIO.PWM(12,50)
    p.start(15)
    def play(p,frequency,tempo):
        p.ChangeFrequency(frequency)
        time.sleep(0.5*tempo)
        
    play(p,333,1)
    time.sleep(0.5)
    play(p,666,1)
    time.sleep(0.5)
    play(p,333,1)
    time.sleep(0.5)
    play(p,666,1)
    time.sleep(0.5)

def PIR():
    List = ['']
    find = 0
    GPIO.setmode(GPIO.BOARD) #Set GPIO pin numbering 

    pir = 26 #Associate pin 26 to pir 

    GPIO.setup(pir, GPIO.IN) #Set pin as GPIO in print "Waiting for sensor to settle"
    first = 0
    time.sleep(2) #Waiting 2 seconds for the sensor to initiate print "Detecting motion" 
    #count=0
    while True: 
        
        if GPIO.input(pir): #Check whether pir is HIGH print "Motion Detected!"
            if(first == 0):
                time.sleep(1.5)
                first = 1
                continue
            print("detected!\n")
            #print(count)
            #count+=1
            break
            time.sleep(3) #D1- Delay to avoid multiple detection
        else:print("not detected!\n")
        time.sleep(0.1) #While loop delay should be less than detection(hardware) delay

def detect(List):
    print(os.getpid(),multiprocessing.current_process().name)
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-master/data/face/haarcascade_eye_tree_eyeglasses.xml')
    userData = json.loads(open('/home/pi/Downloads/opencv-master/data/face/userInfo.txt').read())
    userList = ['']
    BLList = ['']
    find = 0
    for member in userData['User'] :
        userList.append(member["name"])
        BLList.append(member["BTAddress"])
    #print(userList)
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    display_window = cv2.namedWindow("Face")
    time.sleep(0.1)
    last = 0
    id = 0
    frameCount = 0
    #t0 = time.time()
    correctTime = 0
    s=0.5
    #font = cv2.CV_FONT_HERSHEY_COMPLEX
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frameCount+=1
        image = frame.array
        image2 = image.copy()
        #image = cv2.resize(image,None, fx=s,fy=s,interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.1, 3)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                    #print(ex)
                if eyes[0][0] < eyes[1][0]:
                    (x1,y1),(x2,y2) = (eyes[0][0]+eyes[0][2]/2,eyes[0][1]+eyes[0][3]/2),(eyes[1][0]+eyes[1][2]/2,eyes[1][1]+eyes[1][3]/2)
                else:
                    (x2,y2),(x1,y1) = (eyes[0][0]+eyes[0][2]/2,eyes[0][1]+eyes[0][3]/2),(eyes[1][0]+eyes[1][2]/2,eyes[1][1]+eyes[1][3]/2)    
                A = np.zeros((2,3))
                A[0,0:2]=[x2-x1,y2-y1]
                #print('start:')
                #print(A)
                s = (A[0,0]**2+A[0,1]**2)**0.5
                #print(s)
                A[0,0:2]/=s
                A[1,0:2]=[-A[0,1],A[0,0]]
                #print(A)
                A[0,:] *= 50/s
                #print(A)
                A[1,:] *= 95/(h-(y1+y2)/2)
                #print(A)
                A[0,2] = 50-A[0,0]*(x+(x1+x2)/2)-A[0,1]*(y+(y1+y2)/2)
                #print(A)
                A[1,2] = 25-A[1,0]*(x+(x1+x2)/2)-A[1,1]*(y+(y1+y2)/2)
                #print(A)
                #print('end')
                dst_img = cv2.warpAffine(image,A,(100,120))
                temp = cv2.cvtColor(dst_img,cv2.COLOR_BGR2GRAY)
                temp = cv2.resize(temp,(100,100))
                #temp = cv2.equalizeHist(temp)
                total=0
                for row in temp:
                    for value in row:
                         total+=value
                avg = total/10000
                for row in range(len(temp)):
                    for col in range(len(temp[row])):
                        temp[row][col] = ((temp[row][col]/avg)*135)
                id,conf = recognizer.predict(temp)
                #print("------------------")
                #print("ID: "+str(id))
                #print(conf)
                if conf<=1000:
                    print(conf)
                    print(last)
                    print(correctTime+1)
                    
                    cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.putText(image2,userList[id],(x,y+h),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255))
                    if last == id:
                        correctTime+=1
                    elif last == 0 and correctTime == 0:
                        correctTime = 1
                        last = id
                    else:
                        correctTime = 0
                        last = 0
                    
        cv2.imshow("Face",image2)
        #t30 = time.time() - t0
        #print(30*correctTime)
        key = cv2.waitKey(1)
        if frameCount > 50*(correctTime+1):
            cv2.destroyAllWindows()
            return 
        #if int(t30) > 30*(correctTime+1):
            #cv2.destroyAllWindows()
            #return
            #key = ord("q")
        rawCapture.truncate(0)
        if key == ord("q") or correctTime >= 3: # Press 'q' to quit
            for i in List:
                if i == BLList[last]:
                    print(BLList)
                    print("Accept!!!!!!!  "+i)
                    buzzerAccept()
                    find = 1
            if find == 0:
                buzzerFailed_NoPhone()
            break
    List.clear()
    cv2.destroyAllWindows()
    
def scanBeacon(q):
    flag = 0
    print(os.getpid(),multiprocessing.current_process().name)
    userData = json.loads(open('/home/pi/Downloads/opencv-master/data/face/userInfo.txt').read())
    for member in userData['User'] :
        userList.append(member["name"])
        BLUList.append(member["BTAddress"])
    scannerbyDevices = bluetooth.discover_devices(lookup_names = True)
    for addr, name in scannerbyDevices:
        print("Discovered Bluetooth device", addr, "with name:",name)
        if addr in BLUList:
            flag = 1
        List.append(addr)
        
    if flag == 0:
        buzzerFailed_NoPhone()
        print("Do Not search any specific device!")
    q.put(flag)
    q.put(List)
def main():
    # the code of detecting a human
    #someOne =  
    #if someOne == 1:
    time = 0
    q = multiprocessing.Queue()
    PIR()
    print("skip PIR")
    P3 = Process(target = send_leave, args=(q,))
    #P2 = Process(target= detect)
    #P1 = Process(target= scanBeacon, args=(P2,))
    P3.start()
    P2 = Process(target = scanBeacon, args=(q,))
    P2.start()
    flag1 = q.get()
    flag2 = q.get()
    P2.join()
    P3.join()
    List = q.get()
    print("Searched List: {}".format(List))
    if flag1*flag2 == 1:
        detect(List)
        print("Finish!")
    #scanBeacon()Ω
    #detect()
    #P1.start()
    #P2.start()
    #P2.join()
    #P1.join()
    '''
    Thread1 = threading.Thread(target = scanBeacon())
    Thread1.start()
    Thread2 = threading.Thread(target = detect())
    Thread2.start()
    scanner = Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(5.0)

    Thread1.join()
    Thread2.join()
    
    scanner = Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(5.0)
    '''
    
#if __name__ == '__main__':    
while True:        	
    print('main:',os.getpid(),multiprocessing.current_process().name)
    main()
    time.sleep(1)