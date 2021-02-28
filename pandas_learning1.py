# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:15:27 2021

@author: AHAMAD HUSSAIN
"""


def pattern(n):
    k=2*n-2
    for i in range(n):
        for j in range(k):
            print(end=' ')
        k-=1
        for j in range(i+1):
            print('* ',end='')
            
        print('\r')
        #print('\n')

pattern(10)

def full_pyramid(rows):
    for i in range(rows):
        print(' '*(rows-i-1) + '*'*(2*i+1))
        
full_pyramid(6)

def half_pyramid(rows):
    for i in range(rows):
        print( '*'*(2*i+1)+' '*(rows-i-1))
        
half_pyramid(6)

def in_pyramid(rows):
    for i in range(rows):
        print(' '*(rows+i) + '*'*(2*(rows-i)-1))
        
in_pyramid(6)

def pattern(n):
    k=n*2-2
    for i in range(n):
        k-=1
        for j in range(k-2,0,-1):
            print(end=' ')
            print('* ',end='')
            
            
        print('\r')

pattern(5)

def pattern(n):
    k=n-2
    for i in range(n,-1,-1):
        for j in range(k,0,-1):
            print(end=' ')
        k=k+1
        for j in range(i):
            print('* ',end='')  
        print('\r')

pattern(5)

class employee():
    def __init__(self,name,age,id,salary):   //creating a function
        self.name = name // self is an instance of a class
        self.age = age
        self.salary = salary
        self.id = id
 
emp1 = employee("harshit",22,1000,1234) //creating objects
emp2 = employee("arjun",23,2000,2234)
print('emp1.__dict__)//Prints dictionary
      
class square:
    def __init__(self,side):
        self._side=side
        
    @property
    def side(self):
        return self._side
    @side.setter
    def side(self,value):
        if value>=0:
            self._side=value
        else:
            print("error")
    @property
    def area(self):
        return self._side**2
    @classmethod
    def unit_square(cls):
        return cls(1)

s=square(5)
print(s.side)
print(s.area)

from functools import reduce
def a(x,y):
    return x+y
print(reduce(a,[1,2,3,4,5,6,7,8]))

reduce(lambda q,p: q*p, [1,2,3,4,5,6,7,8])

#pip install cv2
import cv2
img1=cv2.imread('Russu.jpg')
imgb=cv2.imread('Russu.jpg',0)
img=cv2.imread('Russu1.jpg')
print(type(img))
print(img.shape)

print(type(imgb))
print(imgb.shape)

cv2.imshow("RushdaInshira", img)
cv2.waitKey(0)
cv2.destroyAllWindows(2000)
 

imgr=cv2.resize(img,(600,600))
cv2.imshow("Rushda Inshira", imgr)
cv2.waitKey(0)
cv2.destroyAllWindows(2000)

imgr=cv2.resize(img,(int(img.shape[1]/9),int(img.shape[0]/9)))
print(imgr.shape)
face_cascade = cv2.CascadeClassifier('C:/Users/Nasre/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('C:/Users/Nasre/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_eye.xml')
cv2.imshow("RushdaInshira", imgr)
cv2.waitKey(0)
cv2.destroyAllWindows(2000)

grey_img=cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(grey_img,scaleFactor=1.1,minNeighbors=5)

face_cascade.load('haarcascade_frontalface_default.xml')
face_cascade.load('C:/Users/Nasre/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

#faces = face_cascade.detectMultiScale(grey_img, 0.9, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(imgr,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = grey_img[y:y+h, x:x+w]
    roi_color = imgr[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#video processing
video=cv2.VideoCapture('video1.mp4')
#help(cv2.VideoCapture)
time.sleep(3)
video.release()
check,frame=video.read()
print(check)
print(frame.shape)
cv2.imshow('Capture',frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
a=1
while True:
    a=a+1
    check,frame=video.read()
    face_cascade = cv2.CascadeClassifier('C:/Users/Nasre/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier('C:/Users/Nasre/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_eye.xml')

    grey_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Capture',rey_img)
    cv2.waitKey(1)
    #if key==ord(q):
       # break
    if a>=10:
        break
print(a)


#11 FEB 2021
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#video from file
import numpy as np
import cv2

cap = cv2.VideoCapture('video1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#to save file
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
#another flavour
import cv2 
import numpy as np 
  
# Creating a VideoCapture object to read the video 
cap = cv2.VideoCapture('video1.mp4') 
  
  
# Loop untill the end of the video 
while (cap.isOpened()): 
  
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, 
                         interpolation = cv2.INTER_CUBIC) 
  
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
  
    # conversion of BGR to grayscale is necessary to apply this operation 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # adaptive thresholding to use different threshold  
    # values on different regions of the frame. 
    Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2) 
  
    cv2.imshow('Thresh', Thresh) 
    # define q as the exit button 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break
  
# release the video capture object 
cap.release() 
# Closes all the windows currently opened. 
cv2.destroyAllWindows() 

#smoothing video
import cv2 
import numpy as np 
  
# Creating a VideoCapture object to read the video 
cap = cv2.VideoCapture('video1.mp4') 
  
  
# Loop untill the end of the video 
while (cap.isOpened()): 
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, 
                         interpolation = cv2.INTER_CUBIC) 
  
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
  
    # using cv2.Gaussianblur() method to blur the video 
  
    # (5, 5) is the kernel size for blurring. 
    gaussianblur = cv2.GaussianBlur(frame, (5, 5), 0)  
    cv2.imshow('gblur', gaussianblur) 
  
    # define q as the exit button 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break
  
# release the video capture object 
cap.release() 
  
# Closes all the windows currently opened. 
cv2.destroyAllWindows() 

#Bitwise operations
# importing the necessary libraries 
import cv2 
import numpy as np 
  
# Creating a VideoCapture object to read the video 
cap = cv2.VideoCapture('video1.mp4') 
  
  
# Loop untill the end of the video 
while (cap.isOpened()): 
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,  
                         interpolation = cv2.INTER_CUBIC) 
  
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
      
    # conversion of BGR to grayscale is necessary to apply this operation 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
  
    # apply NOT operation on image and mask generated by thresholding 
    BIT = cv2.bitwise_not(frame, frame, mask = mask) 
    cv2.imshow('BIT', BIT) 
  
    # define q as the exit button 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break
  
# release the video capture object 
cap.release() 

# Closes all the windows currently opened. 
cv2.destroyAllWindows()

#Edge detection

import cv2
import sys
# The first argument is the image
image = cv2.imread('boxes.jpg')
 
#convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#blur it
blurred_image = cv2.GaussianBlur(gray_image, (7,7), 0)
 
cv2.imshow("Orignal Image", image)
canny = cv2.Canny(blurred_image, 10, 30)
cv2.imshow("Canny with low thresholds", canny)
canny = cv2.Canny(blurred_image, 50, 150)
help(cv2.Canny(blurred_image, 50, 150))
help(cv2.Canny)
cv2.imshow("Canny with high thresholds", canny)
#python edge_detect.py ship.jpg

cv2.imshow("Original image", image)
cv2.imshow("Blurred image", blurred_image)
 
# Run the Canny edge detector
canny = cv2.Canny(blurred_image, 30, 80)
cv2.imshow("Canny", canny)

im, contours, hierarchy= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of objects found = ", len(contours))
