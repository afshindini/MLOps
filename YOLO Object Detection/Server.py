# Import required libraries
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import cvlib as cv
import os


# Define a class for enumerating between available models
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


app = FastAPI(title='Deploying YOLOV3 Model with FastAPI')  # Define an instance of FastAPI class


# Define a GET method for an endpoint
@app.get("/")
def home():
    return "YOLOV3 is deployed. Go to http://localhost:8000/docs for more information."


# Define an endpoint for prediction using the ML model
@app.post("/prediction")
def prediction(model: Model, file: UploadFile = File(...)):
    # Check type of input file
    filepath = file.filename
    if not (filepath.split(".")[-1] in ("jpg", "jpeg", "png")):
        raise HTTPException(status_code=415, detail="Unsupported file extension.")

    # Transform image into opencv image
    rawImage = io.BytesIO(file.file.read())  # Read raw image as bytes
    rawImage.seek(0)  # Start reading image from beginning
    arrayImage = np.asarray(bytearray(rawImage.read()), dtype=np.uint8)  # Convert raw image to numpy array
    image = cv2.imdecode(arrayImage, cv2.IMREAD_COLOR)  # Convert image array to cv2 image

    # Run object detection model and save the results
    detect_object(image, model, filepath)

    # Send the response back to the client
    outputImage = open(f'Results/{filepath}', mode="rb")  # open the saved image
    return StreamingResponse(outputImage, media_type="image/jpeg")


# Function that detects objects and draw a box around them
def detect_object(image, model, filename):
    boundaryBox, label, conf = cv.detect_common_objects(image, model=model)  # Perform object detection

    outputImg = cv.object_detection.draw_bbox(image, boundaryBox, label, conf)  # Create output image
    save_results(outputImg, filename)  # save the output img


# Function for saving the results
def save_results(img, filename):
    # Make directory to save the results
    path = "Results"
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(f'Results/{filename}', img)  # Save the img in the directory


nest_asyncio.apply()    # Allow the server to run in the environment
host = "127.0.0.1"  # Define a host
uvicorn.run(app, host=host, port=8000)  # Run the server
