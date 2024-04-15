import csv
import random
import csv
import random
import numpy as np

"""
With this method you create train and test lists of images and their corresponding labels
Start by reading labels csv (each line: <picture number/name, label>)
In this particular image dataset, there are 102 kinds of flowers labeled 1-102
Note: the terms list and array are used interchangeably throughout
The only reason this works is because the dataset is not very large
"""

def get_train_and_test_lists ():

    with open('example.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Make list of 102 lists
    csv_info = []
    for value in range(0,102): 
        csv_info.append([])

    # print(csv_info)

    # Fill it
    for line in data:
        csv_info[int(line[1])-1].append(int(line[0]))

    # The data is now organized in an array that contains 102 arrays
    # Each individual array in order contain the picture numbers for that label i.e. csv_info[0] has images of label #1
    # Except my labels start at 0 because it's easier to work with and those brits didn't publish a list of labels to flower names so it doesn't matter
    # print("Indices of all images with label = 1 ", csv_info[0])
    # print('Length of this list: ', len(csv_info)) 

    # _____________________________________________

    # Make train and test datasets and labels
    # Declare blank lists
    train_label_list = []
    test_label_list = []
    train_list = []
    test_list = []

    # Fill the lists 
    for value in range(0,102): 
        # randomly assign some to train and some to test - 10% seems good to me
        size_of_test_list = int(round(len(csv_info[value])*0.1))
        temp_test_list = random.sample(csv_info[value], size_of_test_list)

        # Remove test list values from csv list
        for i in csv_info[value][:]:
            if i in temp_test_list:
                csv_info[value].remove(i)

        # Append those suckers to the list
        train_list.extend(csv_info[value])
        test_list.extend(temp_test_list)

        for element in temp_test_list:
            test_label_list.extend([value])
        for element in csv_info[value]:
            train_label_list.extend([value])

    train_label_list = np.array(train_label_list, dtype=int)
    test_label_list = np.array(test_label_list, dtype=int)

    # print(train_list[0:10])
    # print(test_list[0:10])
    # print(train_label_list[0:10])
    # print(test_label_list[0:10])
    # print(len(train_list), len(train_label_list))
    # print(len(test_list), len(test_label_list))
    # print(min(train_list))
    # print(max(train_list))
    # print(min(test_list))
    # print(max(test_list))
    # print(min(test_label_list), max(test_label_list))

    # Convert numbers into filepaths
    for value in range(0,int(len(train_list))):
        train_list[value] = 'image_'+ str(train_list[value]).zfill(5)+'.jpg'
    
    for value in range(0, int(len(test_list))):
        test_list[value]= 'image_'+ str(test_list[value]).zfill(5)+'.jpg'

    # print(train_list[0:10])
    # print(test_list[0:10])
    # print(train_label_list[0:10])
    # print(test_label_list[0:10])

    return train_list, train_label_list, test_list, test_label_list

# get_train_and_test_lists()

