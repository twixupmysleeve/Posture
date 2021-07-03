import csv
import numpy as np
import csv


def get_data():
    output_vectors = open('data/output_vectors.csv')
    input_vectors = open('data/input_vectors.csv')

    output_reader = csv.reader(output_vectors, delimiter=',')
    input_reader = csv.reader(input_vectors, delimiter=',')
    line_count = 0

    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    outputs = []

    for outl in output_reader:
        line_count += 1
        if outl[-1] == 1:
            next(input_reader)
            continue
        inl = next(input_reader)

        # print("#########@#@#@")
        # print(outl)
        # print(inl)
        assert outl[0] == inl[0]
        assert outl[1] == inl[1]

        feature1.append((float(inl[2])))
        feature2.append((float(inl[3])))
        feature3.append((float(inl[4])))
        feature4.append((float(inl[5])))
        feature5.append((float(inl[6])))
        outputs.append((float(outl[2]), float(outl[3]), float(outl[4]), float(outl[5]), float(outl[6]), float(outl[7])))

    print(f'Processed {line_count} lines.')

    return [np.array(feature1), np.array(feature2), np.array(feature3), np.array(feature4), np.array(feature5),
            np.array(outputs)]

import pandas as pd


data = get_data()
mappings = ['c', 'k', 'h', 'r', 'x', 'i']


def discrete(binary_vector, order):
    return "".join([order[i] for i in range(len(binary_vector)) if binary_vector[i] == 1])


dict = {'neck': data[0],
        'knees': data[1],
        'hip': data[2],
        'ankle': data[3],
        'kneey': data[4],
        'target': [discrete(data[5][i], mappings) for i in range(len(data[5]))]}

df = pd.DataFrame(dict)

df.to_csv('dash_data.csv')