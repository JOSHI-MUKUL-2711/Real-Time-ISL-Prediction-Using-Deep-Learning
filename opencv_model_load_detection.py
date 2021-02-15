import cv2
import numpy as np
import torch
import torch.onnx
import os
os.chdir('/home/joshi-mukul/Documents/Models_saved/')
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

onnx_model_path = 'new_onxx_model_V1'
net = cv2.dnn.readNetFromONNX(onnx_model_path)

biggest_pred_index = None
classes = ['0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z']

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    


    contours, hierachy =cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (frame, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
num_frames =0
while True:
    ret, frame = cam.read()

    
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    grayImage = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    HSVImaage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) 


    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(HSVImaage, lowerBoundary, upperBoundary)
    
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(grayImage, grayImage, mask = skinMask)
    
    
    gray_frame = cv2.Canny(skin,60,60)

    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "Waiting for background detection.",(3, 459), cv2.FONT_ITALIC, 1, (0,0,255), 2)
    
    else: 
        hand = segment_hand(gray_frame)
        
        if hand is not None:
            
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
      ROI_top)], -1, (255, 0, 0),1)
            
            thresholded = cv2.resize(thresholded, (128, 128))
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
            cv2.imshow('Image_after_processing',thresholded)

            thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
            try:
                
                blob = cv2.dnn.blobFromImages(thresholded, 1.0,(128,128), swapRB=False, crop=False)
                net.setInput(blob)
                preds = net.forward()

                biggest_pred_index = np.array(preds)[0].argmax()
                cv2.putText(frame_copy, classes[biggest_pred_index],(240, 86), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            except Exception as e:
                cv2.putText(frame_copy, 'None',(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
                break
                

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3)

    num_frames += 1

    cv2.putText(frame_copy, "Real Time Indian Sign Language Detector",
    (10, 20), cv2.FONT_ITALIC, 0.5, (255,255,255), 1)
    cv2.imshow("Sign Detection", frame_copy)



    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()