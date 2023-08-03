# MLOps - Deploying Object Detection Using Yolov3 with fastAPI
The goal of this repository is to deploy a simple object detecion method by using a pretrained models as YOLOv3 and YOLOv3-tiny.
The code is devided into two parts as server and client. The client send appropriate POST requests to server and the server response to those request accordingly. In order to run the code, run server and client parts in two separate terminals.

## Required Libraries
Before running the code Please install the required libraries by:
```
pip install -r requirements.txt
```

## Server Part
The server part uses fastAPI for receiving the POST requests fron the client. The endpoint is considered as http://127.0.0.1/prediction. The handler function is ```prediction``` which receives the requests and the images from client and after applying the YOLOv3 objection detection model on them, it saves them in a ```Result``` folder. Run the server by:
```
python Server.py
```

## Client Part
In the Client part, you make a POSR request to the server and receive the response and save them in a ```Prediction``` folder. In this regard, the ```response``` function is wrtitten for this purpose. Run the client code by:
```
python Client.py
```
