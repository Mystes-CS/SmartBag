import RPi.GPIO as GPIO #Import GPIO library
import time #Import time library 

GPIO.setmode(GPIO.BOARD) #Set GPIO pin numbering 

pir = 26 #Associate pin 26 to pir 

GPIO.setup(pir, GPIO.IN) #Set pin as GPIO in print "Waiting for sensor to settle" 

time.sleep(2) #Waiting 2 seconds for the sensor to initiate print "Detecting motion" 
count=0
while True: 

    if GPIO.input(pir): #Check whether pir is HIGH print "Motion Detected!"
        print("detected!\n")
        print(count)
        count+=1
        time.sleep(3) #D1- Delay to avoid multiple detection
    else:print("not detected!\n")
    time.sleep(0.1) #While loop delay should be less than detection(hardware) delay
