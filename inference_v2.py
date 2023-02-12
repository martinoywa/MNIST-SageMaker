
import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import numpy as np

import requests
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.ModuleDict):
    def __init__(self):
        super(Net, self).__init__()
        #convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)
    
    def forward(self, x):
         # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten image input
        x = torch.flatten(x, 1)
        # add hidden layer, with relu activation function
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

# TODO: Add model_fn
def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    # expects a URI
    uri = json.loads(request_body)["image_uri"]
    logger.info(f"URI: {uri}")
    data = Image.open(requests.get(uri, stream=True).raw).convert('L')
    logger.info(f"DATA: {data}")
    # preprocess image
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
    data = transform(data).unsqueeze(0).to(device)
    logger.info(f"PREPROCESSED DATA: {data}")
    return data


def predict_fn(input_object, model):
    logger.info(f"PREPROCESSED INPUT DATA: {input_object}")
    with torch.no_grad():
        logger.info(f"MAKE PREDICTIONS")
        prediction = model(input_object)
    logger.info(f"PREDICTIONS: {prediction}")
    return prediction


def output_fn(predictions, content_type):
    logger.info(f"OUTPUT FORMATTER")
    assert content_type == 'application/json'
    res = np.argmax(predictions, 1).cpu().numpy().tolist()
    return json.dumps(res)
