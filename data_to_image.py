import csv
import os
import numpy as np
import cv2
from PIL import Image

def data2image():
    '''
    save data to image according to labels
    '''
    
    file_paths = ['datasets/train.csv', 'datasets/test.csv', 'datasets/validation.csv']
    save_paths = ['datasets/train', 'datasets/test', 'datasets/validation']
    for save_path, file_path in zip(save_paths, file_paths):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, (label, pixel) in enumerate(reader):
                image = np.array([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                im = Image.fromarray(image).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                im.save(image_name)


if __name__ == '__main__':
    data2image()
