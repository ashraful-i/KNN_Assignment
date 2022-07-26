# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import sqrt
from PIL import Image

import numpy as np


def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2  # (x1-x2)**2+(y1-y2)**2
    return sqrt(distance)

def get_row_val(test_rows):
    print(len(test_rows))
    r = -1
    g = -1
    b = -1
    for neb in test_rows:
        r, g, b = (1 / len(neb)) * (neb[0]), (1 / len(neb)) * (neb[1]), (1 / len(neb)) * (neb[2])
        #print(neb)
    #print (r, g, b)
    tr = list()
    tr.append(r)
    tr.append(g)
    tr.append(b)
    print(tr)
    return tr

def GetTestAreas(train, test_row1, test_row2):
    print("GN")
    distance1 = list()  # []
    distance2 = list()  # []
    data1 = []
    data2 = []
    for i in train:
        dist1 = Euclidean_distance(test_row1, i)
        dist2 = Euclidean_distance(test_row2, i)
        #print("dist1 = " + str(dist1) + " dist2 = " + str(dist2))
        if(dist1<dist2):
            distance1.append(dist1)
            data1.append(i)
        else:
            distance2.append(dist2)
            data2.append(i)
    print("len")
    print(len(data1))
    print(len(data2))
    print("len end")
    if len(data1) == 0:
        print("data1 0")
        return
    if len(data2) == 0:
        print("data2 0")
        return
    global test_r_1
    global test_r_2
    test_r_1 = get_row_val(data1)
    test_r_2 = get_row_val(data2)
    diff1 = abs(test_r_1[0] - test_row1[0])+abs(test_r_1[1] - test_row1[1])+abs(test_r_1[2] - test_row1[2])
    diff2 = abs(test_r_2[0] - test_row2[0]) + abs(test_r_2[1] - test_row2[1]) + abs(test_r_2[2] - test_row2[2])
    if diff1 < 1 and diff2 < 1:
        print("diff < 1")
        print(diff1, diff2)
        return
    print("End")
    GetTestAreas(train, test_r_1, test_r_2)
    #print(data1)
    #print(data2)

    return data1


def Get_Neighbors(train, test_row, num):
    distance = list()  # []
    data = []
    for i in train:
        dist = Euclidean_distance(test_row, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    """ we are finding index of min distance """
    index_dist = distance.argsort()
    """ we arange our data acco. to index """
    data = data[index_dist]
    """ we are slicing num number of datas """
    neighbors = data[:num]

    return neighbors


# Press the green button in the gutter to run the script.

def update_image(test_row1, test_row2):
    im = Image.open("land.jpg")
    pix = im.load()
    distance1 = list()  # []
    distance2 = list()  # []
    data1 = []
    data2 = []

    for x in range(im_w):
        for y in range(im_h):
            rgb = []
            Blue = pix[x, y][0]
            Green = pix[x, y][1]
            Red = pix[x, y][2]
            rgb.append(Red)
            rgb.append(Green)
            rgb.append(Blue)
            dist1 = Euclidean_distance(test_row1, rgb)
            dist2 = Euclidean_distance(test_row2, rgb)
            if dist1< dist2:
                pix[x, y] = (240, 140, 40)
            else:
                pix[x, y] = (40, 140, 240)

    im.save("knn.jpg")
    im.close()


def predict_classification(train, test_row, num):
    Neighbors = GetTestAreas(train, test_row, num)
    '''
    Classes = []
    for i in Neighbors:
        Classes.append(i[-1])
    prediction = max(Classes, key=Classes.count)
    return prediction
    '''

if __name__ == '__main__':
    im = Image.open("land.jpg")
    pix = im.load()

    im_w = im.size[0]
    im_h = im.size[1]

    dataset = []

    for x in range(im_w):
        for y in range(im_h):
            rgb = []
            Blue = pix[x, y][0]
            Green = pix[x, y][1]
            Red = pix[x, y][2]
            rgb.append(Red)
            rgb.append(Green)
            rgb.append(Blue)
            dataset.append(rgb)
    im.close()
    '''dataset = [
        [7.673756466, 3.508563011, 1],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [2.7810836, 2.550537003, 0]
    ]'''
    print(len(dataset))

    global test_r_1
    global test_r_2
    test_r_1 = [255, 110, 80]
    test_r_2 = [100, 110, 230]
    prediction = predict_classification(dataset, test_r_1, test_r_2)
    print(test_r_1)
    print(test_r_2)
    #print("We expected {}, Got {}".format(dataset[-1][-1], prediction))

    update_image(test_r_1, test_r_2)
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
