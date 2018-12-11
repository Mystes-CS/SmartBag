import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.FisherFaceRecognizer_create()
#detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#path='dataset'
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)
    faces=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
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
faces,ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()