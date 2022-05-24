from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from joblib import load

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", default="decision_tree",
    help="name of the algorithm will work")
ap.add_argument("-d", "--dataset", default="data_nan_mean_no_Island",
    help="path to the dataset")
ap.add_argument("-m", "--model", default="MyProcessModel", 
               help="path to the output model",)
ap.add_argument("-o", "--output", default="./pics/Neural_net/confusion_matrix_normalization.png",
               help="path to the output loss/accuracy plot",)
ap.add_argument("-epoches", "--epochs", default="300", 
               help="path to the picture output")
args = vars(ap.parse_args())

model = ...



X_test = pd.read_csv("./data/" + args["dataset"] + "/X_test.csv").values
y_test = pd.read_csv("./data/" + args["dataset"] + "/y_test.csv").values

y_test = np.squeeze(y_test)
n_values = np.max(y_test) + 1
y_test = np.eye(n_values)[y_test]

if args["algorithm"] == "neural_network":
    L_model = "./results/model/neural_network/" + args["model"]  + "_" + args["epochs"] + "_epochs_" + args["dataset"] + ".hdf5"
    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    model = load_model(L_model)
    # make predictions on the images
    print("[INFO] predicting...")
    results = model.evaluate(X_test, y_test)
elif args["algorithm"] == "decision_tree":
    L_model = "./results/model/decision_tree/" + args["model"] + "_" + args["dataset"] + ".joblib"
    model = load(L_model)

plt.style.use("default")

y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_label = np.argmax(y_test, axis=1)

labels = ["Adelie", "Chinstrap", "Gentoo"]

cm = confusion_matrix(y_test_label, y_predict)
# Normalise
cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(args["output"])
# plt.savefig('./pics/Neural_net/confusion_matrix_normalization.pdf')
plt.show()
