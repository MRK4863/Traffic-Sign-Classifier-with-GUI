from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename 
from keras.models import load_model
import keras
import numpy as np
import pandas as pd
import cv2
import os
from werkzeug.utils import secure_filename

classes = { 1:'Speed limit (20km/h)',
 			2:'Speed limit (30km/h)', 
 			3:'Speed limit (50km/h)', 
 			4:'Speed limit (60km/h)', 
 			5:'Speed limit (70km/h)', 
 			6:'Speed limit (80km/h)', 
 			7:'End of speed limit (80km/h)', 
 			8:'Speed limit (100km/h)', 
 			9:'Speed limit (120km/h)', 
 			10:'No passing', 
 			11:'No passing veh over 3.5 tons', 
 			12:'Right-of-way at intersection', 
 			13:'Priority road', 
 			14:'Yield', 
 			15:'Stop', 
 			16:'No vehicles', 
 			17:'Veh > 3.5 tons prohibited', 
 			18:'No entry', 
 			19:'General caution', 
 			20:'Dangerous curve left', 
 			21:'Dangerous curve right', 
 			22:'Double curve', 
 			23:'Bumpy road', 
 			24:'Slippery road', 
 			25:'Road narrows on the right', 
 			26:'Road work', 
 			27:'Traffic signals', 
 			28:'Pedestrians', 
			29:'Children crossing', 
 			30:'Bicycles crossing', 
 			31:'Beware of ice/snow',
 			32:'Wild animals crossing', 
 			33:'End speed + passing limits', 
 			34:'Turn right ahead', 
 			35:'Turn left ahead', 
 			36:'Ahead only', 
 			37:'Go straight or right', 
 			38:'Go straight or left', 
 			39:'Keep right', 
 			40:'Keep left', 
 			41:'Roundabout mandatory', 
 			42:'End of no passing', 
 			43:'End no passing veh > 3.5 tons' }


root = Tk()

#canvas = Canvas(root, width =500, height =500)
#canvas.pack()

root.title("TRAFFIC SIGN CLASSIFIER")
root.configure( background = '#CDCDCD')
root.geometry("500x500")
root.iconbitmap('ROAD_ICON_1.ico')

text = Label(root, text='UPLOAD THE IMAGE')
text.pack()

global uploaded
#label=Label(root,background='#CDCDCD', font=('arial',15,'bold'))
#sign_image = Label(root)
k=0


def openfilename():
    #progress = Progressbar(root, orient = HORIZONTAL, length =100, mode = 'determinate')
    filepath = askopenfilename()
    return filepath

def model_predict(img_path):
    img = Image.open(img_path)

    #resize the image to a 30x30 with the same strategy as in TM2:
    image_new=img.resize((30,30))
    # Model saved with Keras model.save()
    MODEL_PATH = 'traffic_sign_model_grayscale_2.model'

    # Load your trained model
    model = keras.models.load_model(MODEL_PATH)
    #turn the image into a numpy array
    x=np.array(image_new)
    x_gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x_gray = np.array(x_gray)
    x_gray = x_gray.reshape(1,x_gray.shape[0],x_gray.shape[1],1)
    print(x_gray.shape)
    preds = model.predict(x_gray)
    print('predicted successfully............')
    
    preds = list(preds[0])
            
    pred_class_id = preds.index(max(preds)) +1
    #print(preds)
    #print(pred_class_id)
    return pred_class_id



def upload_img(): 
    global fname
    # Select the Imagename  from a folder  
    fname = openfilename() 
    disp_image()
    print(fname)    

    # opens the image 
    #img = Image.open(x)

def predict_img():
    print('Entered successfully')
    print(fname)
    cid = model_predict(fname)
    
    global class_id
    class_id = cid
    print(classes[cid])
    
    #w = Label(root, text=classes[cid])
    #w.place( x= 200, y= 450)
    text = Text(root, height = 1, width = 40)
    text.insert(INSERT, classes[class_id])
    text.place(x= 50, y= 450)
    
    #text.tag_add("here", "1.0", "1")
    #text.tag_add("start", "1.8", "1.13")
    #text.tag_config("here", background="yellow", foreground="blue")
    #text.tag_config("start", background="black", foreground="green")

def disp_image():
    print('Displaying Image.......')
    photo = ImageTk.PhotoImage(Image.open(fname))
    labelphoto = Label(root, image = photo)
    labelphoto.image = photo
    labelphoto.place(x=50, y=120)
    #w = Canvas(root)
    #w.photo = photo
    #w.place(x=10 , y=120 )



def clear_label_image():
    labelphoto.config(image = "")
    labelphoto = Label(root, image = "")
    
  
upload_button = PhotoImage(file ='interface.png')
btn =  Button(root, image = upload_button, command = upload_img)
btn.place(x=225, y = 40)

btn2 = Button(root, text = "Predict", command = predict_img )
btn2.place(x = 200, y = 420)

#button to clear the label image
btn3 = Button(root, text = "Clear", command=clear_label_image)
btn3.place(x = 300, y = 420)



#classify = 
#label = Label()
root.mainloop()

