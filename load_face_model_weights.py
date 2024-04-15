import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras import Input, layers
from keras.preprocessing import image

import numpy as np

from _get_train_and_test_lists import get_train_and_test_lists
from _list_of_filepaths_to_numpy import list_to_numpy


from tensorflow.keras.models import load_model, Model

my_model = load_model('my_model.keras')

# truncated_model = Model(inputs = my_model.layers[1].input, outputs = my_model.layers[-2].output)  
# predictions = Dense(256, activation = "softmax")(truncated_model.output)
# my_model = Model(inputs = truncated_model.input, outputs = predictions)

my_model.summary()


# ______________________________________________________
# Train model

# Generate datasets and labels
train_list, train_label_list, test_list, test_label_list = get_train_and_test_lists ()

# Convert list of filepaths to numpy array of images
train_numpy = list_to_numpy (train_list)
test_numpy = list_to_numpy(test_list)

# print(train_numpy.shape)
# print(test_numpy.shape)
# print(train_label_list.shape)
# print(test_label_list.shape)

# np.savetxt("listoftrain.csv", train_label_list, fmt='%s')

train_dataset = tf.data.Dataset.from_tensor_slices((train_numpy, train_label_list))
test_dataset = tf.data.Dataset.from_tensor_slices((test_numpy, test_label_list))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

my_model.fit(train_dataset, epochs=2)

# ______________________________________________________
# See if it's any good
# my_model.evaluate(test_dataset)

# preds = my_model.predict(x)
# print('Predictions made')
# print("Raw predictions: ", preds.shape, preds)
# print('Predicted:', decode_predictions(preds))

my_model.save('my_model.keras')


