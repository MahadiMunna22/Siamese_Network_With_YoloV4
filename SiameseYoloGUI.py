from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import time
import pyautogui, sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui as pg
import keyboard

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.fh = 0
        self.stop_Yolo = False
        self.yoloRunning = False
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20 # Interval in ms to get the latest frame
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        self.crop_canvas = tk.Canvas(self.window, width=100, height=100)
        self.crop_canvas.pack(side=LEFT, padx=5, pady=5)

        # Variable for Siamese Network
        self.model = load_model('Weights\weights-improvement-91-0.93.hdf5')
        self.referenceGesture=None
        self.referenceGestureRecordFlag=False
        self.snapShot=False
        
        # Buttons
        
        self.quit = tk.Button(self.window, text= "Quit", command= self.window.destroy)
        self.quit.pack(side=RIGHT, padx=5, pady=5)

        self.hand = tk.Button(self.window, text= "Hand", command= lambda: self.face_hand(1))
        self.hand.pack(side=RIGHT, padx=5, pady=5)

        self.face = tk.Button(self.window, text= "Face", command= lambda: self.face_hand(0))
        self.face.pack(side=RIGHT, padx=5, pady=5)

        self.Compare = tk.Button(self.window, text= "Compare", command= lambda: self.SiameseNetwork(1))
        self.Compare.pack(side=RIGHT, padx=5, pady=5)

        self.StartSN = tk.Button(self.window, text= "Start Siamese Network", command= lambda: self.SiameseNetwork(0))
        self.StartSN.pack(side=RIGHT, padx=5, pady=5)

        self.Compare["state"] = "disable"
        self.StartSN["state"] = "disable"

    def face_hand(self, val):

        self.Compare["state"] = "normal"
        self.StartSN["state"] = "normal"

        if self.yoloRunning == True:
            self.stop_Yolo = True
        self.fh = val
        self.runSiameseYolo()

    def SiameseNetwork(self, val):
        if val == 0:
            self.referenceGestureRecordFlag = not self.referenceGestureRecordFlag
        elif val == 1:
            self.snapShot=True

    def runSiameseYolo(self):

        # fh = 0 -> Hand, fh = 1 -> face
        if self.stop_Yolo == False:
            
            if self.fh == 1:
                net = cv2.dnn.readNet("Weights\yolov4-tiny-hand_best.weights", "cfg_file\yolov4-tiny-testing.cfg")
            elif self.fh == 0:
                net = cv2.dnn.readNet("Weights\yolov4-tiny-nose_best.weights", "cfg_file\yolov4-tiny-testing.cfg")
            
            classes = [""]
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            # colors = np.random.uniform(0, 255, size=(len(classes), 3))
            colors = np.random.uniform(255, 255, size=(len(classes), 3))
            cap = self.cap
            
            ret, frame = cap.read()

            # Display the resulting frame
            frame = cv2.flip(frame, 1)
            img = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
            height, width, channels = img.shape

            # print(img.shape)
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (288, 288), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

                # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        # print("Label ",class_id)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        
                        #pyautogui.moveTo(center_x, center_y)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # print("x,y = "+str(x)+", "+str(y))

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            # print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    try:
                        x, y, w, h = boxes[i]
                        # print("x1,y1 = "+str(x)+", "+str(y))
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        if (label == 1):
                            cv2.circle(img, (x, y), 2, color, 5)
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                        croppedImg = img[y:y+h, x:x+w]
                        # print("Label ",label)
                        croppedImg=cv2.resize(croppedImg,(100,100))
                        croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                        
                        self.crImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2RGB) # to RGB
                        self.crImg = Image.fromarray(self.crImg) # to PIL format
                        self.crImg = ImageTk.PhotoImage(self.crImg)  # to ImageTk format
                        
                        self.crop_canvas.create_image(0, 0, anchor=tk.NW, image=self.crImg)    

                        #Processing it for tensorflow model
                        
                        croppedImg = np.stack((croppedImg,)*3, axis=-1)
                        croppedImg=croppedImg.reshape((1,100,100,3))
                        croppedImg=croppedImg/255
                        if(self.referenceGestureRecordFlag):
                            cv2.putText(img, "Can take snap", (50, 100), font, 3, color, 2)                        
                            self.referenceGesture=None
                            if(self.snapShot==True):
                                self.referenceGesture=croppedImg
                                self.referenceGestureRecordFlag=False
                                self.snapShot = False
                        else:
                            if (self.referenceGesture is None):
                                pass
                            else:                     
                                y=self.model.predict([self.referenceGesture, croppedImg])
                                cv2.putText(img, "Can compare:", (10, 50), font, 3, color, 2)
                                cv2.putText(img, str(y), (10, 80), font, 3, color, 2)
                                print(y)
                    except:
                        pass

            # cv2.imshow("Image", img)
            # Get the latest frame and convert image format
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to RGB
            self.image = Image.fromarray(self.image) # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            
            
            # Update image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            # Repeat every 'interval' ms
            self.window.after(self.interval, self.runSiameseYolo)

            self.stop_Yolo = False
            self.yoloRunning = True
        else:
            self.stop_Yolo = False
            self.yoloRunning = False
            return

                
if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()