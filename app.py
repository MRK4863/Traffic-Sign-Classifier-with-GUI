from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#######################################################################
#from tkinter import *
#from tkinter.ttk import *
from PIL import Image, ImageTk
#from tkinter.filedialog import askopenfilename 
from keras.models import load_model
import keras
import numpy as np
import pandas as pd
import cv2
import os
from werkzeug.utils import secure_filename


#from tensorflow import keras 
from PIL import Image, ImageOps
import cv2 
import random
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

########################################################################


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

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = os.path.join(os.getcwd(), "traffic_sign_model_grayscale_2.model")

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded......... Check http://127.0.0.1:5000/')


def model_predict(model,img_path):
    # message= request.get_json(force=True)
    # encoded=message['image']
    # decoded=base64.b64decode(encoded)
    img = Image.open(img_path)
    
    image_new=img.resize((30,30))
    # Model saved with Keras model.save()
    #MODEL_PATH = 'models/traffic_sign_model_grayscale_2.model'

    # Load your trained model
    #model = keras.models.load_model(MODEL_PATH)
    #turn the image into a numpy array
    x=np.array(image_new)
    x_gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x_gray = np.array(x_gray)
    x_gray = x_gray.reshape(1,x_gray.shape[0],x_gray.shape[1],1)
    print(x_gray.shape)
    preds = model.predict(x_gray)
    print('\n\n\npredicted successfully............\n\n\n')
    
    
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    
    #x = preprocess_input(x, mode='caffe')

    #preds = model.predict(x)
    preds = list(preds[0])
            
    pred_class_id = preds.index(max(preds)) +1
    #print(preds)
    #print(pred_class_id)
    return pred_class_id


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("\nthe current working directory is ..................\n");
        print(os.getcwd())
        print(file_path)
        print("\n\n\n\n\n")

        # Make prediction
        #preds = model_predict(file_path, model)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = preds.index(max(preds)) +1        # ImageNet Decode
        cid = model_predict(model, file_path)
        result = classes[cid]                    # Convert to string
        
        print("\n\nGot the result\n\n")     
        print(result)
        print("\n\n\n\n")

        return result   
    return None


if __name__ == '__main__':
    app.run(debug=True)

