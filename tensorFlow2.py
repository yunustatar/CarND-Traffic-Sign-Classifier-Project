# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#% matplotlib inline

import numpy as np

N = n_classes

ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

training_hist = []
valid_hist = []
test_hist = []
for y in range(0, 43):
    training_hist.append(np.count_nonzero(y_train == y))
    valid_hist.append(np.count_nonzero(y_valid == y))
    test_hist.append(np.count_nonzero(y_test == y))

fig, ax = plt.subplots()
rects1 = ax.bar(ind, training_hist, width, color='r')

rects2 = ax.bar(ind + width, valid_hist, width, color='y')

rects3 = ax.bar(ind + 2 * width, test_hist, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('Number of samples in set')
ax.set_title('Number of samples')
ax.set_xticks(ind[::2])
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Training', 'Validation', 'Test'))

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import numpy as np
X_train = np.array(X_train,dtype=np.float32)
y_train = np.array(y_train,dtype=np.int32)

X_valid = np.array(X_valid,dtype=np.float32)
y_valid = np.array(y_valid,dtype=np.int32)

X_test = np.array(X_test,dtype=np.float32)
y_test = np.array(y_test,dtype=np.int32)

def preprocess(input_imgs):
    gry = np.zeros((len(input_imgs),32,32,1),dtype=np.float32)
    X = np.array(input_imgs)
    r = X[:, :, :, 1]
    g = X[:, :, :, 2]
    b = X[:, :, :, 0]
    gry = (0.299 * r  + 0.587 * g + 0.114 * b - 128.0) / 128.0
    return gry

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
print(type(X_train))
print(X_train.shape)
X_train_p = preprocess(X_train)
X_valid_p = preprocess(X_valid)
X_test_p = preprocess(X_test)

from sklearn.utils import shuffle

X_train_p, y_train = shuffle(X_train_p, y_train)

### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf

EPOCHS = 5
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten


def LeNet(x, kp):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x64.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 64), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x32. Output = 14x14x64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # dropout
    conv1 = tf.nn.dropout(conv1, kp)

    # Layer 2: Convolutional. Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 32), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 800.
    fc0 = flatten(conv2)

    # dropout
    fc0 = tf.nn.dropout(fc0, kp)

    # Layer 3: Fully Connected. Input = 800. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 400), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 400. Output = 200.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(400, 200), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(200))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, kp)

    # Layer 5: Fully Connected. Input = 84. Output = 43 (n_classes).
    fc3_W = tf.Variable(tf.truncated_normal(shape=(200, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


from tensorflow.contrib.layers import flatten


def LeNet1111(x, kp):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # dropout
    conv1 = tf.nn.dropout(conv1, kp)

    # Layer 2: Convolutional. Output = 10x10x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # dropout
    fc0 = tf.nn.dropout(fc0, kp)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 800), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(800))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(800, 200), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(200))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, kp)

    # Layer 5: Fully Connected. Input = 84. Output = 43 (n_classes).
    fc3_W = tf.Variable(tf.truncated_normal(shape=(200, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))
y_classes = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y_classes, n_classes)
keep_prob = tf.placeholder(tf.float32)

rate = 0.001

logits = LeNet(x_inputs, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
