import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from keras.preprocessing import image


import numpy as np
import cv2

from _get_train_and_test_lists import get_train_and_test_lists
from _list_of_filepaths_to_numpy import list_to_numpy


# print('Starting program__________________________________________________________________________________________________________________________________________________________________________________________________')


def decode_predictions(preds, top=5):
    LABELS = None
    if len(preds.shape) == 2:
        if preds.shape[1] == 2622:
            fpath = get_file('rcmalli_vggface_labels_v1.npy',
                             V1_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        elif preds.shape[1] == 8631:
            fpath = get_file('rcmalli_vggface_labels_v2.npy',
                             V2_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        else:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                             '(samples, 8631) for V2.'
                             'Found array with shape: ' + str(preds.shape))
    else:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                         '(samples, 8631) for V2.'
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[str(LABELS[i].encode('utf8')), pred[i]] for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"
    # print('___________________________________________________________________')
    # print("Made it to resnet_identity_block", input_tensor.shape)
    # print("filters: ", filters, "kernel size:" , kernel_size)

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    # print("Finished resnet_identity_block", x.shape)
    # print('___________________________________________________________________')
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    # print('___________________________________________________________________')
    # print("Made it to resnet_conv_block", input_tensor.shape)
    # print("filters: ", filters, "kernel size:" , kernel_size)
    bn_axis = 3
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    # print("Next Step:",x.shape)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "bn")(x)
    # print("Next Step:",x.shape)
    x = Activation('relu')(x)
    # print("Next Step:",x.shape)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    # print("Next Step:",x.shape)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "bn")(x)
    # print("Next Step:",x.shape)
    x = Activation('relu')(x)
    # print("Next Step:",x.shape)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    # print("Next Step:",x.shape)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "bn")(x)
    # print("Next Step:",x.shape)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    # print("Next Step: shortcut",shortcut.shape)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "bn")(
        shortcut)
    # print("Next Step: shortcut",shortcut.shape)

    x = layers.add([x, shortcut])
    # print("Next Step:",x.shape)
    x = Activation('relu')(x)
    # print("Finished resnet_conv_block", x.shape)
    # print('___________________________________________________________________')

    return x


def RESNET50(include_top=True, weights='vggface',
             pooling=None,
             classes=102):
         
    img_input = Input(shape=(224, 224, 3))


    # print('___________________________________________________________________')
    # print("Starting the model:",type(img_input),img_input.shape)

    bn_axis = 3

    x = Conv2D(64, (7,7), use_bias=False, strides=(2, 2), padding='same')(img_input)
    # print("Next Step:",x.shape)
    x = BatchNormalization(axis=bn_axis)(x)
    # print("Next Step:",x.shape)
    x = Activation('relu')(x)
    # print("Next Step:",x.shape)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # print("Next Step:",x.shape)

    # print('Finished first layers of model', x.shape)
    # print('___________________________________________________________________')


    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # print('___________________________________________________________________')
    # print("Last layers: ", x.shape)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # print("average pooling:",x.shape)

    x = Flatten()(x)
    # print("flatten:",x.shape)

    x = Dense(classes, activation='softmax', name='classifier')(x)
    # print("dense:",x.shape)
    # print('Finished model', x.shape)
    # print('___________________________________________________________________')


    # Create model.
    model = Model(img_input, x)

    return model

print('_____________________________________________________________________________________________________________________________________________________________')

# ______________________________________________________

# Build model
my_model = RESNET50()
my_model.summary()

# ______________________________________________________
# Train model

# Generate datasets and labels from csv file
train_list, train_label_list, test_list, test_label_list = get_train_and_test_lists ()

# Convert list of filepaths to numpy array of the actual images - this function calls the preprocess function 
train_numpy = list_to_numpy (train_list)
test_numpy = list_to_numpy(test_list)

# Test a bunch of stuff
# print(train_numpy.shape)
# print(test_numpy.shape)
# print(train_label_list.shape)
# print(test_label_list.shape)
# print(train_label_list[0:10])

# Save the image categories and labels
np.savetxt("trainlabels.csv", train_label_list, fmt='%s')
np.savetxt("testlabels.csv", test_label_list, fmt='%s')
np.savetxt("trainimages.csv", train_list, fmt='%s')
np.savetxt("testimages.csv", test_list, fmt='%s')

train_dataset = tf.data.Dataset.from_tensor_slices((train_numpy, train_label_list))
test_dataset = tf.data.Dataset.from_tensor_slices((test_numpy, test_label_list))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

my_model.fit(train_dataset, epochs=35)


# See if it's any good
# my_model.evaluate(test_dataset)


# ______________________________________________________

# Test model
# preds = my_model.predict(x)
# print('Predictions made')
# print("Raw predictions: ", preds.shape, preds)
# print('Predicted:', decode_predictions(preds))

# ______________________________________________________
# Save it 

my_model.save('my_model.keras')


