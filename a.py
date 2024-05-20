#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from flask import Flask, request, jsonify
import numpy as np
import json
import cv2
import random
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

# app = Flask(__name__)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Define the necessary classes and functions

def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)

class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels, kernel_size=self._kernel_shape, stride=self._stride, padding=0, bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3], name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3], name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3), name=name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(480, [192, 96, 208, 16, 48, 64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(512, [160, 112, 224, 24, 64, 64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(512, [128, 128, 256, 24, 64, 64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(512, [112, 144, 288, 32, 64, 64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(528, [256, 160, 320, 32, 128, 128], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(832, [256, 160, 320, 32, 128, 128], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(832, [384, 192, 384, 48, 128, 128], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=1024, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        x = x.mean(2)
        return x

# Load the model and set it to evaluation mode
model = InceptionI3d(num_classes=100, in_channels=3)
model.load_state_dict(torch.load('I3D_100.pt', map_location=torch.device('cpu')))
model.eval()

# Load the label map
# with open('dataloader/isl_classInd.json', 'r') as f:
#     label_map = json.load(f)
label_map = {}

# Read the label.txt file
with open('100_labels.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            key = int(parts[0])
            value = parts[1]
            label_map[key] = value

# print(label_map)

# Define the Flask routes
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        predictions = predict_sign(filepath)
        return jsonify(predictions)

# def predict_sign(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (224, 224))
#         frames.append(frame)
#     cap.release()
#     frames = np.array(frames)
#     frames = video_to_tensor(frames).float()
#     frames = frames.unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(frames)
#         _, preds = torch.max(outputs, 1)
#         pred_class = preds[0].item()
#         pred_label = label_map[str(pred_class + 1)]
    
#     return {'label': pred_label, 'confidence': torch.nn.functional.softmax(outputs, dim=1)[0, pred_class].item()}


def predict_sign(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = video_to_tensor(frames).float()
    frames = frames.unsqueeze(0)

    with torch.no_grad():
        outputs = model(frames)
        _, preds = torch.max(outputs, 1)
        pred_class = preds[0].item()
        pred_label = label_map[pred_class]
    
    return {'label': pred_label, 'confidence': torch.nn.functional.softmax(outputs, dim=1)[0, pred_class].item()}


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Handle POST request for prediction here
#     # Example: Parse input data, perform prediction, and return response
#     data = request.json
#     # Perform prediction using the data
#     prediction = {}  # Replace this with your actual prediction logic
#     return jsonify(prediction)

# Route for prediction
# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded video file from the request
    video_file = request.files['video']
    
    # Save the uploaded video file to a specified path
    video_path = 'uploads/' + video_file.filename
    video_file.save(video_path)
    
    # Perform prediction using the uploaded video file
    prediction = predict_sign(video_path)
    
    # Return the prediction as a JSON response
    # pass
    return jsonify(prediction)

    


# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app

# if __name__ == '__main__':
#     app.run(debug=True)
