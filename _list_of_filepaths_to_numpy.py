from _preprocess_image import preprocess_image
import numpy as np

def list_to_numpy (list_of_img_names):
    big_array = []
    for img in list_of_img_names:
        big_array.append(preprocess_image(img))

    # Make numpy array
    big_array = np.array(big_array)

    return big_array