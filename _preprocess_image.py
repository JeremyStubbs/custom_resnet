from tensorflow.keras import backend as K
import numpy as np
import cv2


def preprocess_input(x):
    x_temp = x.astype('float64')
    np.expand_dims(x_temp, axis=0)
    

    x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 91.4953
    x_temp[..., 1] -= 103.8827
    x_temp[..., 2] -= 131.0912

    x_temp = K.constant(x_temp)
    # print("Preprocessing Done:",type(x_temp), x_temp.shape)

    return x_temp

def preprocess_image(filepath):
    image_path = '.\\Your_images\\'+filepath
    img = cv2.imread(image_path) 

    # Might as well do preprocessing here
    # You have more control over it by resizing and cropping the image here
    img_shape = img.shape
    # print(img_shape)
    width = img_shape[0]
    length = img_shape[1]

    if width > length:
        width = int(round(width/length*224))
        length = 224
        # print('width bigger,', 'new width = ', width, 'new length =', length)
        img = cv2.resize(img, (length, width))
        # print('new image dimensions =' , img.shape, length, width)
        img = img[0:224, 0:224]
        # print('new image dimensions =' , img.shape)
    else:
        length = int(round(length/width*224))
        width = 224
        # print('length bigger, ', 'new width =', width, 'new length =', length)
        img = cv2.resize(img, (length, width))
        # print('new image dimensions =' , img.shape, length, width)
        img = img[0:224, 0:224]
        # print('new image dimensions =' , img.shape)

    # img = preprocess_input(img)
        

    # cv2.imshow('image',img)
    # cv2.waitKey(0) 
    return img
    
# x = preprocess_image('image_00001.jpg')
# print(x.shape)



# Can do the same thing with built in tensorflow functions but then you have to rely on their image processing
# from keras.preprocessing import image
# img = image.load_img('.\\102_flowers\\jpg\\image_00001.jpg', target_size=(224, 224),keep_aspect_ratio = True)
# img = image.img_to_array(img)

