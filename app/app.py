import os
import sys
from flask import Flask, render_template, request
import numpy as np
import cv2
from base64 import b64decode, b64encode
from cvzone.HandTrackingModule import HandDetector
from pyzbar.pyzbar import decode
import tensorflow as tf
from tensorflow import keras

#model = tf.keras.models.load_model('app/keras_model.h5')
#model = tf.keras.models.load_model('./app/my_model.h5', compile=False)

class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        self.model = tf.keras.models.load_model(self.model_path)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        # resize the image to a 224x224 with the same strategy as in TM2:
        imgS = cv2.resize(img, (224, 224))
        # turn the image into a numpy array
        image_array = np.asarray(imgS)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        self.data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal

maskClassifier = Classifier('app/keras_model.h5', 'app/labels.txt')

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', user_image = '')
    else:

        return render_template('index.html', user_image = '')
    
@app.route("/api/checkanswer", methods=['GET','POST'])
def check_answer():
    txt64 = request.form.get("todo")
    encoded_data = txt64.split(',')[1]
    encoded_data = b64decode(encoded_data)
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    crop_img = img[15:165, 45:195]
    items =['A','B','NA']
    
    prediction,index = maskClassifier.getPrediction(crop_img, scale=1, draw= False) 
    
    cv2.putText(crop_img, str(items[index]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
        
    _, im_arr = cv2.imencode('.png', crop_img)
    im_bytes = im_arr.tobytes()
    im_b64 = b64encode(im_bytes).decode("utf-8")
    return str(items[index])
