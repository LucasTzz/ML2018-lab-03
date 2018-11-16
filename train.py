import cv2
import os
import numpy as np
import sys
import sklearn
sys.path.append('/Users/taozizhuo/PycharmProjects/experiment/venv/ML2018-lab-03')
from ensemble import AdaBoostClassifier
import feature
import pickle
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    def get_features(path):

        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        features = []
        # read the images and extract the features
        for image_path in image_paths:
            img = cv2.imread(image_path)
            # convert into gray image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_reshape = cv2.resize(gray_img, (25, 25), interpolation=cv2.INTER_CUBIC)
            image = feature.NPDFeature(img_reshape)
            pre_features = feature.NPDFeature.extract(image)
            AdaBoostClassifier.save(pre_features, "save.p")
            face_feature = AdaBoostClassifier.load("save.p")
            features.append(face_feature)
        return features

    # establish numpy arrays
    Faces = np.array(get_features('./datasets/original/face'))
    Non_faces = np.array(get_features('./datasets/original/nonface'))
    n_samples, n_features = Faces.shape
    y_faces = []
    y_non_faces = []
    # set labels
    for i in range(n_samples):
        y_faces.append(1)
        y_non_faces.append(-1)
    y_faces = np.array(y_faces).reshape((-1, 1))
    y_non_faces = np.array(y_non_faces).reshape((-1, 1))
    # split training set and validation set
    Faces_train, Faces_val, y_faces_train, y_faces_val = train_test_split(Faces, y_faces, test_size=0.2)
    Non_faces_train, Non_faces_val, y_non_faces_train, y_non_faces_val = train_test_split(Non_faces, y_non_faces,

                                                                                          test_size=0.2)
    # put pictures with and without faces together to form the training set and validation set
    X_train = np.concatenate((Faces_train, Non_faces_train), axis=0)
    X_val = np.concatenate((Faces_val, Non_faces_val), axis=0)
    y_train = np.concatenate((y_faces_train, y_non_faces_train), axis=0)
    y_val = np.concatenate((y_faces_val, y_non_faces_val), axis=0)

    X_train = np.column_stack((y_train, X_train))
    np.random.shuffle(X_train)
    y_train = X_train[:, 0]
    X_train = np.delete(X_train, 0, axis=1)
    # set a initial classifier
    classifier = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(), 10)
    # train
    classifier.fit(X_train, y_train)
    predicts = classifier.predict(X_val)
    # get the accuracy and write report to a txt file
    report = sklearn.metrics.classification_report(y_true=y_val, y_pred=predicts, digits=2)
    f = open('classifier_report.txt', 'w')
    f.write(report)
    f.close()
    pass
