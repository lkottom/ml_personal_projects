from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageOps
import re
import pickle

app = Flask(__name__)

# Load weights and biases
with open('weights_biases.pkl', 'rb') as f:
    weights_biases = pickle.load(f)

W1 = weights_biases['W1']
b1 = weights_biases['b1']
W2 = weights_biases['W2']
b2 = weights_biases['b2']

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def forward_prop(W1, b1, W2, b2, X):
    # print(W1.shape)
    # print(X.shape)
    # Now we do the linear transformation (the combines the weight and input tensor using the dot product and the weight)
    # The connecting part of the neurons is done by matrix multiplication (Weight tensor x X tensor...)
    # print(f"W1 shape: {W1.shape}")
    # print(f"X shape: {X.shape}")
    Z1 = W1.dot(X) + b1 #Z1 is the activation here
    # print(Z1[0])
    # ReLU helps us get the output of the of the neurons within the hidden layer 
    A1 = ReLu(Z1)
    # print(A1[0])
    # Now, we have to connect the neurons from the hidden layer to the output layer, this is done by taking another dot product and adding bias
    Z2 = W2.dot(A1) + b2
    # Now, we get the output of neurons in the output layer, since we want probablilies, we use the softmax equations (due to this being a classification problem)
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
    return exp_Z / sum_exp_Z


def get_predictions(A2):
    A2_max = np.argmax(A2, axis=0)
    return A2_max

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    img_str = re.search(r'base64,(.*)', data_url).group(1)
    img_bytes = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28))

    img = ImageOps.invert(img)
    
    img.save("debug_image.png")

    img_array = np.array(img).astype(np.float32).reshape(784, 1)
    img_array /= 255.0  # Normalize the image

    _, _, _, A = forward_prop(W1, b1, W2, b2, img_array) 
    print(A)
    prediction = get_predictions(A)
    print(prediction)
    
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)