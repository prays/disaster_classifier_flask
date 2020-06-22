#Importing neccessary packages and libraries
from __future__ import division, print_function

#Flask
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf


from keras.applications.imagenet_utils import decode_predictions
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.backend import set_session

import numpy as np
import os

K.clear_session()

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

model = load_model('disaster_and_non_model.h5')

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model._make_predict_function()

#FLASK INIT
app = Flask(__name__)

#APP CONFIGURATIONS
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['UPLOAD_FOLDER'] = 'images'

app.config['SECRET_KEY'] = '6575fae36288be6d1bad40b99808e37f'

def prepare_image(path):
    """
    This function returns the numpy array of an image
    """
    img = image.load_img(path, target_size=(200, 200))

    img_array = image.img_to_array(img)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    img_array_expanded_dims /= 255.

    #img_array_expanded_dims = np.array(img_array_expanded_dims)

    return img_array_expanded_dims

def decode(labels):
    if labels == 0:
        return 'cyclone'
    elif labels == 1:
        return 'earthquake'
    elif labels == 2:
        return 'flood'
    elif labels == 3:
        return 'wildfire'
    else:
        return 'no disaster'


@app.route('/', methods=['GET', 'POST'])

def predict():

    if request.method == 'POST':

        if request.files:

            img = request.files['file']

            extension = img.filename.split('.')[1]

            if extension not in ['jpeg', 'png', 'jpg']:
                flash('File format not suported.', 'warning')

            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename))

                img.save(file_path)

                preprocessed_image = prepare_image(file_path)

                global sess
                global graph
                with graph.as_default():
                    set_session(sess)
                    predictions = model.predict_classes(preprocessed_image)

                    label = decode(predictions)

                    if label == 'no disaster':

                        flash('It\'s safe, {} detected.'.format(label), 'success')

                    else:
                        flash('EVACUATE NOW! It\'s a {}. Call 112 now for help.'.format(label), 'danger')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
