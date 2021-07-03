import csv
import numpy as np


def get_data():
    output_vectors = open('data/output_vectors.csv')
    input_vectors = open('data/input_vectors.csv')

    output_reader = csv.reader(output_vectors, delimiter=',')
    input_reader = csv.reader(input_vectors, delimiter=',')
    line_count = 0

    outputs = []
    inputs = []

    for outl in output_reader:
        if outl[-1] == 1:
            next(input_reader)
            continue
        inl = next(input_reader)

        # print("#########@#@#@")
        # print(outl)
        # print(inl)
        assert outl[0] == inl[0]
        assert outl[1] == inl[1]

        outputs.append((float(outl[0]), float(outl[1]), float(outl[2]), float(outl[3]), float(outl[4])))
        inputs.append((float(inl[0]), float(inl[1]), float(inl[2]), float(inl[3]), float(inl[4])))

    print(f'Processed {line_count} lines.')

    return np.array(inputs), np.array(outputs)
