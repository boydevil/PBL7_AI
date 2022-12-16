from keras.models import load_model
import keras.applications.xception as xception
from PIL import Image, ImageOps
import numpy as np
import warnings
import requests
from io import BytesIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

warnings.filterwarnings("ignore", category=FutureWarning)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model.h5", custom_objects={"xception": xception}, compile=False)
# Load the labels
class_names = open('labels.txt',).readlines()
IMAGE_WIDTH = 71
IMAGE_HEIGHT = 71
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32)

def predict(URL):
    # Replace this with the path to your image
    # URL = 'https://res.cloudinary.com/' + USER_ID + '/image/upload/PBL6/' + IMG_ID
    response = requests.get(URL)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32)) 

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = str(class_names[index])
    confidence_score = str(prediction[0][index]*100)

    return class_name, confidence_score