import cv2
import mediapipe as mp
import pytesseract
import pyttsx3

import numpy as np
import os


#to get colour palette images
#=====================================================
color_img_folder = "D:\LUMNR_PYTHON\Computer_Vision\Mediapipe_Projects\project_virtual_painter\paint_colors"
color_imgs = os.listdir(color_img_folder)
print(color_imgs)
color_img_lst = []

for img in color_imgs:
    image = cv2.imread(f'{color_img_folder}/{img}')
    color_img_lst.append(image)

#print(color_img_lst)

#=======================================================
#to get default top image

top_img = color_img_lst[3]
print(top_img.shape)
draw_color = (255,255,255) #dafault draw colour

#=========================================================

#setting webcam

cap = cv2.VideoCapture(0)

# to get the supported frame resolutions
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Supported resolutions:")
print(f"Width: {frame_width}")
print(f"Height: {frame_height}")

'''output:
Supported resolutions:
Width: 640.0
Height: 480.0'''
#=========================================================

#to detect hand land marks, creating objects for the classes
det_hand = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = det_hand.Hands()

#==========================================================

xp,yp = 0,0
brush_thickness = 4
eraser_thickness = 50

img_canvas = np.zeros((480,640,3),np.uint8)

#=========================================================================================

#C:\Program Files\Tesseract-OCR\tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

text_speech = pyttsx3.init() #init() is a class of pyttsx3. we creat an obj text_speech

#============================================================================================
 
while True:

    success,img = cap.read()
    img = cv2.flip(img,1)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    img_bgr = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
    
    

    lmlist = []
    tipids = [4,8,12,16,20]

    index_tip,middle_tip = 8,12  #tip points of index fingers and middle finger
#=========================================================================================================
#detecting hand landmarks

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            draw.draw_landmarks(img_bgr,handlms,det_hand.HAND_CONNECTIONS,draw.DrawingSpec(color = (170,212,255),thickness = 1,circle_radius = 1),draw.DrawingSpec(color = (170,206,255),thickness = 1))

            for id,lm in enumerate (handlms.landmark):
                #print(id,lm)
                h,w,c = img_bgr.shape
                cx = int(lm.x*w)   #to get pixel points
                cy = int(lm.y*h)
                lmlist.append([id,cx,cy])

                if id == index_tip:
                    cx_index = int(lm.x*w)
                    cy_index = int(lm.y*h)
                    
                elif id == middle_tip:
                    cx_middle = int(lm.x*w)
                    cy_middle = int(lm.y*h)
                    
                    
    #print(lmlist)
   
#========================================================================================================
#finding the position of fingers

    fingerlist = []

    if len(lmlist) != 0 and len(lmlist) == 21:

    #Handling Thumb
        if lmlist[12][1] < lmlist[20][1]:     #for checking right hand

            if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]: #right hand -- x value of tip increases when finger closed
                fingerlist.append(0)  #append 0 when finger closed
            else:
                fingerlist.append(1)  #append 1 when finger open

        else:                                #for checking left hand
            if lmlist[tipids[0]][1] < lmlist[tipids[0] - 1][1]:  #left hand -- x value of tip decreases when finger closed
                fingerlist.append(0)
            else:
                fingerlist.append(1)

         #for other fingers   
        for id in range(1,5):  
            if lmlist[tipids[id]][2] > lmlist[tipids[id] - 2][2]:
                fingerlist.append(0)
            else:
                fingerlist.append(1)

        print(fingerlist)

#==========================================================================================================

        # for selecting colors - index and middle fingers are up

        if fingerlist[1] == 1 and fingerlist[2] == 1: #index and middle finger positions
            
            xp,yp = 0,0  #updatingting the drawing starting pont
            cv2.rectangle(img_bgr,(cx_index,cy_index),(cx_middle,cy_middle),draw_color,-1)
            print('SELECTION MODE')

        # colour selections
            if cy_index < 80:
                if 140< cx_index <240:
                    top_img = color_img_lst[4]
                    draw_color = (0,0,255)
                elif 240< cx_index <340:
                    top_img = color_img_lst[2]
                    draw_color = (0,255,0)
                elif 340< cx_index <440:
                    top_img = color_img_lst[0]
                    draw_color = (255,0,0)
                elif 440< cx_index <540:
                    top_img = color_img_lst[5]
                    draw_color = (0,255,255)
                elif 540 < cx_index < 640:
                    top_img = color_img_lst[1]
                    draw_color = (0,0,0)

            cv2.rectangle(img_bgr,(cx_index,cy_index),(cx_middle,cy_middle),draw_color,-1)
        
        #for drawing - index finger is up

        if fingerlist[1] == 1 and fingerlist[2] == 0:
            cv2.circle(img_bgr,(cx_index,cy_index),15,draw_color,-1)
            print('DRAWING MODE')

            if xp == 0 and yp == 0:  #in the first frame co-ordinates are zero. so it is updating to
                xp,yp = cx_index,cy_index  #co -ordinates of index finger

            if draw_color == (0,0,0): #whenever select eraser 
                cv2.line(img_bgr,(xp,yp),(cx_index,cy_index),draw_color,eraser_thickness)
                cv2.line(img_canvas,(xp,yp),(cx_index,cy_index),draw_color,eraser_thickness)

            else:  #whenever select other colours
                cv2.line(img_bgr,(xp,yp),(cx_index,cy_index),draw_color,brush_thickness)
                cv2.line(img_canvas,(xp,yp),(cx_index,cy_index),draw_color,brush_thickness)

            xp,yp = cx_index,cy_index   #updating values
          
#=========================================================================================================                 
    
    #blending the images

    img_grey = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _,img_inv = cv2.threshold(img_grey,10,255,cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

    img_bgr = cv2.bitwise_and(img_bgr,img_inv)
    img_bgr = cv2.bitwise_or(img_bgr,img_canvas)

#==========================================================================================================
    
    img_bgr[0:80,0:640] = top_img #height,width -- adding color palette img to the frame
    #img_bgr[80:480,0:640] = np.full((400,640,3),255,np.uint8)

    #img_bgr = cv2.addWeighted(img_bgr,0.5,img_canvas,0.5,0)

#==========================================================================================================
   
   #displaying all the images

    cv2.imshow('Canvas',img_canvas)
    cv2.imshow('Inv',img_inv)
    cv2.imshow('Virtual Paint',img_bgr)
    
#==============================================================================================================
   
    if cv2.waitKey(1) & 0XFF == ord('X'):
       
        #to get text from output image

        output_path = r'D:\LUMNR_PYTHON\Computer_Vision\Mediapipe_Projects\project_virtual_painter\img_canvas.jpg'
        cv2.imwrite(output_path,img_canvas)

        img_out = cv2.imread(output_path)

        data = pytesseract.image_to_data(img_out,output_type = pytesseract.Output.DICT) 
        print(data)

        img_new = np.full((480,640,3),255,np.uint8)
        img_new[0:80,0:640] = top_img

        words = [i for i in (data['text']) if i != '']
        print(words)
        y = 200
        for i in words:
            
            cv2.putText(img_new,i,(100,y),fontFace = cv2.FONT_HERSHEY_COMPLEX,fontScale = 2,color = (0,255,0),thickness = 2)
            
            #to get speech from text
            text = i
            text_speech.say(text) #calling a fn say using the obj text_speech
            text_speech.runAndWait()
            y = y + 60
        
        
        cv2.imshow('out',img_new)
        cv2.waitKey(2000) 
#======================================================================================================================================        
    #if cv2.waitKey(1) & 0XFF == ord('w'):
        #break
    if cv2.waitKey(1) & 0XFF == ord('Z'):
        break
cap.release()
cv2.destroyAllWindows()

