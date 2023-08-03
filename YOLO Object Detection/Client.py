# Import libraries
import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display
import glob


# Make a POST request to the server and return the response
def response(url, imagefile, verbose=True):
    files = {'file': imagefile}
    res = requests.post(url, files=files)
    statusCode = res.status_code
    if verbose:
        msg = "Everything goes well" if statusCode == 200 else "There is an error while handling the request"
        print(msg)
    return res


# display image received from the server response
def save_response_image(res, filename):
    if not os.path.exists("Predictions"):
        os.mkdir("Predictions")

    imageStream = io.BytesIO(res.content)
    imageStream.seek(0)
    file_bytes = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite(f'Predictions/{filename}', image)
    display(Image(f'Predictions/{filename}'))


baseUrl = 'http://localhost:8000'
endpoint = '/prediction'
model = 'yolov3-tiny'
fullUrl = baseUrl + endpoint + "?model=" + model

path = glob.glob(os.path.join("data", "*.jpg"))  # Path to all files in data folder
for file in path:
    with open(file, "rb") as imageFile:
        prediction = response(fullUrl, imageFile, verbose=False)
    save_response_image(prediction, file.split("/")[-1])