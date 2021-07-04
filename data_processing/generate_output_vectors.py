import csv
import cv2


def get_total_frames(video):
    filename = "data/processed/" + video + "_squat.mp4"
    cap = cv2.VideoCapture(filename)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames


rows = []

fps = 12

with open('./data/labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for r in csv_reader:
        row = r[0].split()
        rows.append((row[0], row[1], row[2]))
        line_count += 1
    print(f'Processed {line_count} lines.')


file = open("./data/output_vectors.csv", "w")
frame_number = 0
for row in rows:
    if "end" in row[1]:
        end_frame = int(get_total_frames(row[0]))
    else:
        end_frame = int(float(row[1]) * fps)

    for i in range(frame_number, end_frame):
        c = 1 if "c" in row[2] else 0
        k = 1 if "k" in row[2] else 0
        h = 1 if "h" in row[2] else 0
        r = 1 if "r" in row[2] else 0
        x = 1 if "x" in row[2] else 0
        i = 1 if "i" in row[2] else 0

        frame_number += 1

        line = "{},{},{},{},{},{},{},{}\n".format(row[0], frame_number, c, k, h, r, x, i)
        file.write(line)
    file.flush()

    if "end" in row[1]:
        frame_number = 0

file.close()
