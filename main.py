import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf

vido_cap = cv2.VideoCapture(0)
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

classifier = Classifier("Modal/keras_model.h5", "Modal/labels.txt")


folder = "Data\C"
counter = 0

#lables for output
labels = ["A", "B", "C","D","E","FUCK", "I LOVE YOU"]

while True:
    success , img = vido_cap.read()
    imgSizes = 300
    offset = 20

    hands, img = detector.findHands(img, draw=True, flipType=True)
    imgOutput = img.copy()    

    # Check if any hands are detected
    if hands:
        

        
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        #center1 = hand1['center']  # Center coordinates of the first hand
        

        x,y,w,h = hand1['bbox']
        imgWhite = np.ones((imgSizes, imgSizes,3), np.uint8)*255
        


        #croped image
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] 
        

        imageCropShape = imgCrop.shape
        


        aspectRatio =  h / w

        if aspectRatio>1:
            k = imgSizes / h
            wCal = math.ceil(k*w)
            imagResize = cv2.resize(imgCrop, (wCal,imgSizes))
            imageResizShape = imagResize.shape
            wGap = math.ceil(imgSizes- wCal)/2
            #imgWhite[0:imageResizShape[0] , 0:imageResizShape[1]] = imagResize
            imgWhite[:, int(wGap):int(wCal + wGap)] = imagResize
            predictions , index = classifier.getPrediction(imgWhite)
            print(predictions, index)
        
        else:
            k = imgSizes / w
            hCal = math.ceil(k*h)
            imagResize = cv2.resize(imgCrop, (imgSizes,hCal))
            imageResizShape = imagResize.shape
            hGap = math.ceil(imgSizes- hCal)/2
            imgWhite[int(hGap):int(hCal + hGap), :] = imagResize
            predictions , index = classifier.getPrediction(imgWhite)


        cv2.putText(imgOutput, labels[index], (x+45,y-20) , cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)   


        #cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imgWhite_BG",imgWhite)

        


        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

        # Calculate distance between specific landmarks on the first hand and draw it on the image
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),scale=10)    



    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)

 