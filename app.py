import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

import argparse
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", default="decision_tree",
    help="name of the algorithm will work")
ap.add_argument("-d", "--dataset", default="data_nan_mean_no_Island",
    help="path to the dataset")
ap.add_argument("-m", "--model", default="MyProcessModel",
    help="path to the output model")
ap.add_argument("-o", "--output", default="confusion_matrix_normalization", 
               help="path to the picture output")
ap.add_argument("-batch", "--batch", default="16", 
               help="number of batch")
ap.add_argument("-epoches", "--epochs", default="300", 
               help="number of epochs")
args = vars(ap.parse_args())


save_model = args["model"]
save_output = args["output"]
model = ...
# prepare data
X_train = pd.read_csv('./data/' + args["dataset"] + '/X_train.csv').values
y_train = pd.read_csv('./data/' + args["dataset"] + '/y_train.csv').values
X_test = pd.read_csv('./data/' + args["dataset"] + '/X_test.csv').values
y_test = pd.read_csv('./data/' + args["dataset"] + '/y_test.csv').values

y_train = np.squeeze(y_train)
n_values = np.max(y_train) + 1
y_train = np.eye(n_values)[y_train]

y_test = np.squeeze(y_test)
n_values = np.max(y_test) + 1
y_test = np.eye(n_values)[y_test]

if args["algorithm"] == "neural_network":
    save_model = "./results/model/neural_network/" + args["model"]  + "_" + args["epochs"] + "_epochs_" + args["dataset"] + ".hdf5"
    save_output = "./pics/Neural_net/" + args["output"] + "_" + args["epochs"] + "_epochs_" + args["dataset"] + ".png" 
    batch_size=int(args["batch"])
    epochs=int(args["epochs"])
    input_shape=(9,)
    
    if(args["dataset"] == "data_nan_mean_no_Island"):
        input_shape=(8,)

    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax', name='output'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


    model.fit(X_train, y_train, verbose=2, batch_size=batch_size, epochs=epochs)
    model.save(save_model)
    results = model.evaluate(X_test, y_test)
elif args["algorithm"] == "decision_tree":
    save_model = "./results/model/decision_tree/" + args["model"] + "_" + args["dataset"] + ".joblib"
    save_output = "./pics/decision_tree/" + args["output"] + "_" + args["dataset"] + ".png"
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # save model
    dump(model, save_model)


plt.style.use('default')

y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_label = np.argmax(y_test, axis=1)

labels = ["Adelie", "Chinstrap", "Gentoo"]

# cm = confusion_matrix(y_test_label, y_predict)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# disp.plot(cmap=plt.cm.Blues)
# plt.savefig('./pics/Neural_net/confusion_matrix')
# plt.savefig('./pics/Neural_net/confusion_matrix.pdf')
# plt.show()

cm = confusion_matrix(y_test_label, y_predict)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(save_output)
# plt.savefig('./pics/Neural_net/confusion_matrix_normalization.pdf')
plt.show()