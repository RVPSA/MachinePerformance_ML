import random
import time

import numpy
import sklearn
import pickle
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Sets random seed to increase repeatability
random.seed(23)
numpy.random.seed(23)

def current_ms() -> int:
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)

if __name__ == "__main__":
    """
    Main of the data analysis
    """

    # load dataset PANDAS / NUMPY
    my_dataset = read_csv("labeled_dataset.csv")
    label_obj = my_dataset["label"]
    data_obj = my_dataset.drop(columns=["label", "time", "datetime"])
    # the row above is "equivalent" to
    # my_dataset.drop(columns=["label"], inplace=True)

    # split dataset
    train_data, test_data, train_label, test_label = \
        train_test_split(data_obj, label_obj, test_size=0.2)

    # choose classifier SCIKIT LEARN
    # Set of classifiers that I want to run and compare
    classifier = VotingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                ('nb', GaussianNB()),
                                                ('dt', DecisionTreeClassifier())])



    before_train = current_ms()
    classifier = classifier.fit(train_data, train_label)
    after_train = current_ms()

    # Testing the trained model
    predicted_labels = classifier.predict(test_data)
    end = current_ms()

    # Computing metrics to understand how good an algorithm is
    accuracy = sklearn.metrics.accuracy_score(test_label, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(test_label, predicted_labels).ravel()
    print("%s: Accuracy is %.4f, train time: %d, test time: %d TP: %d, TN: %d, FN: %d, FP: %d" % (
    classifier.__class__.__name__, accuracy, after_train - before_train, end - after_train, tp, tn, fn, fp))

    # Save the trained model as a pickle file
    model_filename = "trained_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(classifier, file)

    print(f"Model saved as {model_filename}")
