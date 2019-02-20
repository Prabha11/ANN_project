# -*- coding: utf-8 -*-
"""
Created on Sun Jan 06 09:25:14 2019
@author: PRABHA
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2

number_of_features = 19201


# Reading the dataset
def read_dataset():
    df = pd.read_csv("gestures.csv")

    X = df[df.columns[0:number_of_features-1]].values
    y1 = df[df.columns[number_of_features-1]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)

    return X, Y, y1


# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encod = np.zeros((n_labels, n_unique_labels))
    one_hot_encod[np.arange(n_labels), labels] = 1
    return one_hot_encod


# Read the dataset
X, Y, y1 = read_dataset()

# Define the important parameters and variable to work with the tensors
learning_rate = 0.3
training_epochs = 50
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print('n_dim', n_dim)
n_class = 2
model_path = 'Modle/ANN_modle'

# Define the hidden layers

n_hidden_1 = 100

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# Define the model
def multilayer_perceptron(x, weights, biases):

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# Define the weights and the biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Call modle
y = multilayer_perceptron(x, weights, biases)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)

print(prediction)
print(X[150].reshape(1, number_of_features-1))

cap = cv2.VideoCapture(0)

# for i in range(100000):
while True:

    ret, frame_original = cap.read()
    frame = cv2.GaussianBlur(frame_original, (5, 5), 0)
    frame = cv2.pyrDown(frame)
    frame = cv2.pyrDown(frame)

    canny = cv2.Canny(frame, 100, 150)
    # print(canny.shape)
    ret, canny = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    # print(canny.shape)

    image_array = np.array(canny)
    image_array[image_array > 0] = 1
    flatten_array = image_array.ravel()
    flatten_list = list(flatten_array)

    # x_df = pd.
    # print(flatten_list)

    cv2.imshow("Show", frame_original)

    prediction_run = sess.run(prediction, feed_dict={x: [flatten_list]})
    print('pred:', int(prediction_run))

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
