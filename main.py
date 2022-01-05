from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import asksaveasfile
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

root = Tk()
root.geometry("540x500")
root.resizable(width=False, height=False)
root.configure(bg='#BEB2A7')

Label(root,
         text="Image PreProcessing Models",
         fg="blue",
         bg="yellow",
         font="Helvetica 21 bold italic").pack()
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


def openi():
    global x
    x = openfn()
    # image
    global image
    # Load the input image
    image = cv2.imread(x)

def viewimage():
    try:
        cv2.imshow('Original', image)
    except:
            messagebox.showwarning("Warning", "First Select an image")



def bgr():
    try:
        #cv2.imshow('Original', image)

        # Use the cvtColor() function to grayscale the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('BGR', rgb_image)
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterbgr.png",rgb_image)
        else:
            return
        # Window shown waits for any key pressing event
        cv2.destroyAllWindows()
        
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")

def binary():
    try:
        img=cv2.imread(x,2)
        ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # converting to its binary form
        bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binary", bw_img)
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterbinary.png",bw_img)
        else:
            return

        
        cv2.destroyAllWindows()
        
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")
def grayscale():
    try:

        # Use the cvtColor() function to grayscale the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Original', image)
        cv2.imshow('GRAYSCALE', gray_image)
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\aftergrayscale.png",gray_image)
        else:
            return
        # Window shown waits for 
        
        cv2.destroyAllWindows()
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")
def hsv():
    try:

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
        #cv2.imshow('Original image',image)
        cv2.imshow('HSV image', hsvImage)
   
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterhsv.png",hsvImage)
        else:
            return
        
        cv2.destroyAllWindows()
  
        
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")
def rgb():  
    try:


        # Use the cvtColor() function to grayscale the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Original', image)
        cv2.imshow('RGB', rgb_image)
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterrgb.png",rgb_image)
        else:
            return
        # Window shown waits for any key pressing event
        cv2.destroyAllWindows()
       
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")

def ycrcb():
    try:

        YCrCbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    
        #cv2.imshow('Original image',image)
        cv2.imshow('YCrCb image', YCrCbImage)
    
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterycrcb.png",YCrCbImage)
        else:
            return
        
        cv2.destroyAllWindows()
        
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")


def yuv():
    try:
        yuvImage = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
        #cv2.imshow('Original image',image)
        cv2.imshow('YUV image', yuvImage)
    
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afteryuv.png",yuvImage)
        else:
            return
        
        cv2.destroyAllWindows()
        
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")

def hsi():
    try:
        image = cv2.imread(x)
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")     
    hsi_lwpimg=RGB_TO_HSI(image)
    #cv2.imshow ("rgb_lwpimg", image)
    cv2.imshow ("HSI", hsi_lwpimg)
    key=cv2.waitKey(0)
    answer=messagebox.askquestion("askquestion","do you want to save")
    if answer=='yes':
        cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterhsi.png",hsi_lwpimg)
    else:
        return

    cv2.destroyAllWindows()
    

def RGB_TO_HSI(im):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(im)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi

def hsl():
    try:
        image = cv2.imread(x)
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")   

    hsi_lwpimg=RGB_TO_HSL(image)
    #cv2.imshow ("rgb_lwpimg", image)
    cv2.imshow ("HSL", hsi_lwpimg)
    key=cv2.waitKey(0)
    answer=messagebox.askquestion("askquestion","do you want to save")
    if answer=='yes':
        cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afterhsl.png",hsi_lwpimg)
    else:
        return
    
    cv2.destroyAllWindows()
    

def RGB_TO_HSL(im):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(im)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_lightness(red, blue, green):
            return np.divide(blue + green + red, 1)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_lightness(red, blue, green)))
        return hsi

def cmyk():
    try:
        image = cv2.imread(x)
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")
    hsi_lwpimg=RGB_TO_CMYK(image)
    #cv2.imshow ("rgb_lwpimg", )
    cv2.imshow ("CMYK", hsi_lwpimg)
    key=cv2.waitKey(0)
    answer=messagebox.askquestion("askquestion","do you want to save")
    if answer=='yes':
        cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\aftercmyk.png",hsi_lwpimg)
    else:
        return
    #if key == ord ("q"):
    cv2.destroyAllWindows()
    

def edge():
    try:

        edgeImage = cv2.Canny(image,100,200)
    
        #cv2.imshow('Original image',image)
        cv2.imshow('edge image', edgeImage)
    
        cv2.waitKey(0)
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afteredge.png",edgeImage)
        else:
            return
        
        cv2.destroyAllWindows()
       
    except NameError:
            messagebox.showwarning("Warning", "First Select an image")

def edgeColor():
    try:
        # convert to RGB
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert to grayscale
        gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        #gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image1 = cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)
        plt.imshow(image1)
        plt.show()
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afteredgeColor.png",image1)
        else:
            return
        
        cv2.destroyAllWindows()
        
    except NameError:
        messagebox.showwarning("Warning", "First Select an image")

def edgeColorD():
    try:
        # convert to grayscale
        gray = cv2.Canny(image,300,200)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image1 = image.copy()
        cv2.drawContours(image=image1, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('EDGE', image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        answer=messagebox.askquestion("askquestion","do you want to save")
        if answer=='yes':
            cv2.imwrite("C:\\Users\\vnaik\\Desktop\\project\\afteredgeColor1.png",image1)
        else:
            return
        
        cv2.destroyAllWindows()
        
    except NameError:
        messagebox.showwarning("Warning", "First Select an image")

def RGB_TO_CMYK(im):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(im)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_lightness(red, blue, green):
            return np.divide(blue + green + red, 1)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            red=1-red/255
            blue=1-blue/255
            green=1-green/255
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_lightness(red, blue, green)))
        return hsi

btn1=Button(root, text='Select Image',font="Helvetica 11 bold italic",bg='#ffffff',fg='#000000',height=1,width=10,command=openi).place(x=125,y=55)

btn1=Button(root, text='View image',font="Helvetica 11 bold italic",bg='#ffffff',fg='#000000',height=1,width=10,command=viewimage).place(x=325,y=55)

btn= Button(root, text='RGB',font="Helvetica 12 bold italic",bg='#3a7bd5',fg='#ffffff',height=3,width=8,command=rgb).place(x=25,y=100)

btn = Button(root, text='Binary',font="Helvetica 12 bold italic",bg='#348AC7',fg='#ffffff',height=3,width=8, command=binary).place(x=25,y=200)

btn = Button(root, text='Grayscale',font="Helvetica 12 bold italic",bg='#24a0ed',fg='#ffffff',height=3,width=8, command=grayscale).place(x=125,y=100)

btn = Button(root, text='HSV',font="Helvetica 12 bold italic",bg='#2bc0e4',fg='#ffffff',height=3,width=8, command=hsv).place(x=125,y=200)

btn = Button(root, text='BGR',font="Helvetica 12 bold italic",bg='#00d2ff',fg='#ffffff',height=3,width=8, command=bgr).place(x=225,y=100)

btn = Button(root, text='YCrCb',font="Helvetica 12 bold italic",bg='#3a7bd5',fg='#ffffff',height=3,width=8, command=ycrcb).place(x=225,y=200)

btn = Button(root, text='YUV',font="Helvetica 12 bold italic",bg='#348AC7',fg='#ffffff',height=3,width=8, command=yuv).place(x=325,y=100)

btn = Button(root, text='HSI',font="Helvetica 12 bold italic",bg='#24a0ed',fg='#ffffff',height=3,width=8, command=hsi).place(x=325,y=200)

btn = Button(root, text='HSL',font="Helvetica 12 bold italic",bg='#2bc0e4',fg='#ffffff',height=3,width=8, command=hsl).place(x=425,y=100)

btn = Button(root, text='CMYK',font="Helvetica 12 bold italic",bg='#348AC7',fg='#ffffff',height=3,width=8, command=cmyk).place(x=425,y=200)

btn = Button(root, text='Edge Detection (B&W)',font="Helvetica 12 bold italic",bg='#00AB66',fg='#ffffff',height=3,width=22, command=edge).place(x=30,y=300)

btn = Button(root, text='Edge Detection (COLOR1)',font="Helvetica 12 bold italic",bg='#00AB66',fg='#ffffff',height=3,width=22, command=edgeColor).place(x=280,y=300)

btn = Button(root, text='Edge Detection (COLOR2)',font="Helvetica 12 bold italic",bg='#00AB66',fg='#ffffff',height=3,width=22, command=edgeColorD).place(x=170,y=400)

root.mainloop()