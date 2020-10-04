import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app

app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('covid19_xray_resnet_50.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def convert_format(path):
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  from skimage.transform import resize
  image = cv2.imread(path,1)
  image=image/255.0
  resized = resize(image, (224,224))
  return resized.reshape((-1,224,224,3))

def model_predict(img_path, model):
    preds = model.predict(convert_format(img_path))
    print(preds)
    if preds[0]>=0.5:
        return('Covid-19')
    else:
        return('noncovid')

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

        # Make prediction

        preds = model_predict(file_path, model)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=False)
