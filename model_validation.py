import os
import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as sp

def search_data(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise IOError('The file path "{}" is not existed!'.format(file_path))
    faces = {}
    for dirpath, subdirs, filenames in os.walk(file_path):
        for image in (file for file in filenames if file.endswith('.jpg')):
            path = os.path.join(dirpath, image)
            label = dirpath[-1]
            if label not in faces:
                faces[label] = []
            faces[label].append(path)

    return faces


def load_validation_data(faces):
    codec = sp.OneHotEncoder(sparse=False, dtype=float)
    x, y = [], []
    for label, filenames in faces.items():
        for filename in filenames:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x.append(gray.ravel())
            y.append([float(label) + 1.])

    x = np.array(x, dtype=float)
    y = codec.fit_transform(np.array(y, dtype=float))

    return x, y
    

def load_model(val_x, val_y_, prob):
    
    ckpt = tf.train.get_checkpoint_state('models/')
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y_ = graph.get_tensor_by_name('y_:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')

        return sess.run(accuracy, feed_dict={x: val_x, y_: val_y_, keep_prob: prob})


if __name__ == '__main__':
    validation_faces = search_data('datasets/validation')
    validation_x, validation_y_ = load_validation_data(validation_faces)
    accuracy = load_model(validation_x, validation_y_, 0.5)
    print('validation accuracy:', accuracy)
    
