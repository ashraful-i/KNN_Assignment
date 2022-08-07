# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import sqrt
from PIL import Image
import pandas as pd
import numpy as np
import sys

def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2  # (x1-x2)**2+(y1-y2)**2
    return sqrt(distance)

def get_row_val(test_rows):
    print(len(test_rows))
    r = 0
    g = 0
    b = 0
    for neb in test_rows:
        r += neb[0]
        g += neb[1]
        b += neb[2]

    r = r / len(test_rows)
    g = g / len(test_rows)
    b = b / len(test_rows)
    # print(neb)
    # print (r, g, b)
    tr = list()
    tr.append(r)
    tr.append(g)
    tr.append(b)
    print(tr)
    return tr

def get_row_val_hr(test_rows, tr_set):
    #print(test_rows)
    age = 0
    anaemia = 0
    creatinine_phosphokinase = 0
    diabetes = 0
    ejection_fraction = 0
    high_blood_pressure = 0
    platelets = 0
    serum_creatinine = 0
    serum_sodium = 0
    sex = 0
    smoking = 0
    time = 0
    for cnt, items in enumerate(test_rows):
        print(cnt)
        for item in items:
            age += item.age
            anaemia += item.anemia
            creatinine_phosphokinase += item.creatinine_phosphokinase
            diabetes += item.diabetes
            ejection_fraction += item.ejection_fraction
            high_blood_pressure += item.high_blood_pressure
            platelets += item.platelets
            serum_creatinine += item.serum_creatinine
            serum_sodium += item.serum_sodium
            sex += item.sex
            smoking += item.smoking
            time += item.time

        age = age/len(items)
        anaemia = anaemia/len(items)
        datafrm = {'age': age, }


def GetTestAreas(dataset, n, distance, train_set):
    #print(train)
    data = [0 for x in range(n)]
    dist = [0 for x in range(n)]

    #print(distance)
    #print(data)

    for cnt_dataset, i in dataset.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_trainset, j in train_set.iterrows():
            tr_lst = [j.age, j.anaemia, j.creatinine_phosphokinase, j.diabetes, j.ejection_fraction,
                      j.high_blood_pressure, j.platelets, j.serum_creatinine, j.serum_sodium, j.sex, j.smoking, j.time]
            df_lst = [i.age, i.anaemia, i.creatinine_phosphokinase, i.diabetes, i.ejection_fraction,
                      i.high_blood_pressure, i.platelets, i.serum_creatinine, i.serum_sodium, i.sex, i.smoking, i.time]
            dis_temp = Euclidean_distance(tr_lst, df_lst)
            if dis_temp<min_dist:
                min_dist = dis_temp
                min_idx = cnt_trainset
        distance[cnt_dataset][min_idx] = i

    test_rows = [[] for x in range(n)]

    for cnt_row,row in enumerate(distance):
        for cnt, item in enumerate(row):
            if type(item) == int:
                continue
            #print(cnt, item)
            distance[cnt_row][cnt] = item
            test_rows[cnt].append(item)
    print (len(test_rows[0]))
    print(len(test_rows[1]))
    print(len(test_rows[2]))
    get_row_val_hr(test_rows, train_set)
    #global test_r_1
    #test_r_1 = get_row_val(data1)
    '''
    global test_r_1
    test_r_1 =  [0 for x in range(n)]
    for cnt, item in enumerate(test_r_1):
        test_r_1[cnt] = get_row_val(distance[])
    test_r_2 = get_row_val(data2)
    test_r_3 = get_row_val(data3)
    diff1 = abs(test_r_1[0] - test_row1[0]) + abs(test_r_1[1] - test_row1[1]) + abs(test_r_1[2] - test_row1[2])
    diff2 = abs(test_r_2[0] - test_row2[0]) + abs(test_r_2[1] - test_row2[1]) + abs(test_r_2[2] - test_row2[2])
    diff3 = abs(test_r_3[0] - test_row3[0]) + abs(test_r_3[1] - test_row3[1]) + abs(test_r_3[2] - test_row3[2])
    if diff1 < 1 and diff2 < 1 and diff3 < 1:
        print("diff < 1")
        return
    print("End")
    '''
    #GetTestAreas(train, test_r_1, test_r_2, test_r_3)
    # print(data1)
    # print(data2)


def GetTestAreas3(train, test_row1, test_row2, test_row3):
    print("GN")
    distance1 = list()  # []
    distance2 = list()  # []
    distance3 = list()  # []
    data1 = []
    data2 = []
    data3 = []
    for i in train:
        dist1 = Euclidean_distance(test_row1, i)
        dist2 = Euclidean_distance(test_row2, i)
        dist3 = Euclidean_distance(test_row3, i)
        # print("dist1 = " + str(dist1) + " dist2 = " + str(dist2))
        if (dist1 < dist2 and dist1 < dist3):
            distance1.append(dist1)
            data1.append(i)
        elif (dist3 < dist2):
            distance3.append(dist3)
            data3.append(i)
        else:
            distance2.append(dist2)
            data2.append(i)
    if len(data1) == 0:
        print("data1 0")
        return
    if len(data2) == 0:
        print("data2 0")
        return
    if len(data3) == 0:
        print("data3 0")
        return
    global test_r_1
    global test_r_2
    global test_r_3
    test_r_1 = get_row_val(data1)
    test_r_2 = get_row_val(data2)
    test_r_3 = get_row_val(data3)
    diff1 = abs(test_r_1[0] - test_row1[0]) + abs(test_r_1[1] - test_row1[1]) + abs(test_r_1[2] - test_row1[2])
    diff2 = abs(test_r_2[0] - test_row2[0]) + abs(test_r_2[1] - test_row2[1]) + abs(test_r_2[2] - test_row2[2])
    diff3 = abs(test_r_3[0] - test_row3[0]) + abs(test_r_3[1] - test_row3[1]) + abs(test_r_3[2] - test_row3[2])
    if diff1 < 1 and diff2 < 1 and diff3 < 1:
        print("diff < 1")
        return
    print("End")
    GetTestAreas3(train, test_r_1, test_r_2, test_r_3)
    # print(data1)
    # print(data2)


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
    im_w, im_h = 100, 100
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
            if dist1 < dist2:
                pix[x, y] = (240, 140, 40)
            else:
                pix[x, y] = (40, 140, 240)

    im.save("knn.jpg")
    im.close()


def update_image3(test_row1, test_row2, test_row3):
    im = Image.open("land2.jpg")
    pix = im.load()
    distance1 = list()  # []
    distance2 = list()  # []
    data1 = []
    data2 = []
    im_w, im_h = 100, 100
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
            dist3 = Euclidean_distance(test_row3, rgb)
            if dist1 < dist2 and dist1 < dist3:
                pix[x, y] = (40, 140, 240)
            elif dist2 < dist3:
                pix[x, y] = (10, 200, 20)
            else:
                pix[x, y] = (240, 140, 40)

    im.save("knn.jpg")
    im.close()


if __name__ == '__main__':
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',', header=[0])
    df = df.sample(n=df.shape[0], random_state=10)

    #print(df.shape[0])
    mask = int(df.shape[0] * 80 / 100)
    #print(mask)

    training_data = df[:mask]
    testing_data = df[mask:]
    # print(f"No. of training examples: {training_data.shape[0]}")
    # print(f"No. of testing examples: {testing_data.shape[0]}")

    # print(training_data)
    # print(testing_data)
    # df = df.reindex(index=np.roll(df.index, df.shape[0]-mask)).reset_index(drop=True)
    # print(df)
    # mask = np.random.rand(len(df)) = < 0.8
    num_of_set = 3
    distance = [[-1 for x in range(num_of_set)] for y in range(len(training_data))]  # []

    #print (distance)

    train_set = training_data.sample(n=num_of_set, random_state=10).reset_index(drop=True)
    training_data = training_data.reset_index(drop=True)
    # print(train_set.columns)
    GetTestAreas(training_data, num_of_set, distance, train_set)
    '''
    print(test_r_1)
    print(test_r_2)
    # print("We expected {}, Got {}".format(dataset[-1][-1], prediction))

    update_image3(test_r_1, test_r_2, test_r_3)
    '''
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
