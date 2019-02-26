import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# [['color', 'shape', 'size'], label]


color = ['red', 'blue', 'green']
shape = ['cube', 'cone', 'sphere']
size = ['large', 'medium', 'small']


def generator(i):
    random.seed(i)
    color_index1 = random.randint(0,2)
    shape_index1 = random.randint(0,2)
    size_index1 = random.randint(0,2)
    # color1 = color[color_index1]
    # shape1 = shape[shape_index1]
    # size1 = size[size_index1]
    color_index2 = random.randint(0, 2)
    shape_index2 = random.randint(0, 2)
    size_index2 = random.randint(0, 2)
    # color2 = color[color_index2]
    # shape2 = shape[shape_index2]
    # size2 = size[size_index2]
    label = 0
    if size_index1 >= size_index2:
        label = 1
    results = [color_index1, shape_index1, size_index1, color_index2, shape_index2, size_index2, label]
    return results


def model_training(x_train, x_test, y_train, y_test):
    #################################### __init__  ####################################
    model = Sequential()
    model.add(Dense(64, input_dim=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    ####################################################################################

    # print(x_train)
    # print(y_train)

    ####################################################################################
    # loop managed by agent
    for i in range(0, 20):
        # fit be in its own method '''learn''', training ipt, training opt, num_epoch = 1
        # batch_size = size of ipt
        model.fit(x_train, y_train,
                  epochs=1,
                  batch_size=128)
    ####################################################################################

    # leave score out
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)
    return model


if __name__ == '__main__':
    result = []
    for i in range(0, 1000):
        sample = generator(i)
        result.append(sample)

    np.savetxt('inputs.txt', result, delimiter= ',')
    result = np.array(result)
    # print(result.shape)
    y = result[:,6]
    # print(y)
    x = np.delete(result, 6, axis=1)
    # print(x)

    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.30)
    model = model_training(x_train_, x_test_, y_train_, y_test_)

    cus_color_1 = color.index(input("color for 1st item"))
    cus_shape_1 = shape.index(input("shape for 1st item"))
    cus_size_1 = size.index(input("size for 1st item"))
    cus_color_2 = color.index(input("color for 2nd item"))
    cus_shape_2 = shape.index(input("shape for 2nd item"))
    cus_size_2 = size.index(input("size for 2nd item"))
    cus_input = [[cus_color_1, cus_shape_1, cus_size_1, cus_color_2, cus_shape_2, cus_size_2]]
    # cus_input = [[1,1,1,1,1,1], [1,2,3,2,3,1],[2,3,2,3,2,1],[2,3,1,3,2,3],[2,3,3,3,2,1],[1,2,2,3,2,1],[2,1,1,2,2,2]]
    cus_input = np.array(cus_input)
    ###################################### __call__ ######################################
    # dict of ipt features - activation, convert to ipt
    predictions = model.predict(cus_input)
    predictions.flatten()
    # covert to opt
    ######################################################################################
    # print(predictions)
    for prediction in predictions:
        p = round(prediction[0])
        if int(p) == 1:
            print("It fits")
        else:
            print("It does't fit")
        # print(round(prediction[0]))

    # print(round(prediction[0]))
