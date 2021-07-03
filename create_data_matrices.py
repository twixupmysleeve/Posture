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
            input_reader.next()
            continue
        inl = input_reader.next()
        assert outl[0] == inl[0]
        assert outl[1] == inl[1]

        outputs.append((outl[0], outl[1], outl[2], outl[3], outl[4]))
        inputs.append((inl[0], inl[1], inl[2], inl[3], inl[4]))

    print(f'Processed {line_count} lines.')

    return np.array(inputs), np.array(outputs)
