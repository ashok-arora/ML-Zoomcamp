#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

interpreter = tflite.Interpreter(model_path='dogs_cats.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


classes = [
           'dog',
           'cat'
]

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    prepared_img = prepare_image(img, (150, 150))
    x = np.array(prepared_img, dtype='float32')

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context=None):
    url = event['url']
    result = predict(url)
    return result