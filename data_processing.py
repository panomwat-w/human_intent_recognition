import os
import json
import numpy as np

fname = "dataset/dataset.json"

def report_class():
    data = json.load(open(fname))
    label_count = {}
    for motion in data:
        label = motion["label"]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    print(label_count)

if __name__ == '__main__':
    report_class()