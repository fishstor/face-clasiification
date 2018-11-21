import tensorflow as tf
import numpy as np
import cv2


def load_model(val_x, prob):
    
    ckpt = tf.train.get_checkpoint_state('models/')
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y_ = graph.get_tensor_by_name('y_:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        f_out = graph.get_tensor_by_name('f_out:0')

        return sess.run(f_out, feed_dict={x: val_x, keep_prob: prob})


def face_capture_and_recognition():

    catograies = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    v = cv2.VideoCapture(0)
    face_recognizer = cv2.CascadeClassifier('haar/face.xml')

    while True:
        frame = v.read()[1]
        faces = face_recognizer.detectMultiScale(frame, 1.2, 5)
        
        for x_left, y_top, width, height in faces:
            face = frame[ y_top:y_top + height, x_left:x_left + width]
            resized_face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_NEAREST)

            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            reshape_face = gray_face.reshape(1, -1)
            catogray = load_model(reshape_face, 0.5)
            max_arg = np.argmax(catogray.ravel())
            emotion = catograies[max_arg]

            cv2.rectangle(frame, (x_left, y_top), (x_left + width, y_top + height), (0, 255, 0))
            cv2.putText(frame, emotion, (x_left, y_top - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            cv2.imshow('video', frame)

        if cv2.waitKey(25) == 27:
            break


    v.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_capture_and_recognition()
