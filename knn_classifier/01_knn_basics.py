#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as met
from termcolor import colored


# Read our dataset
df = pd.read_csv("../datasets/iris_pandas.csv")

# Extract features
features = df.columns[:4].tolist()

# ako zelimo da izdvojimo odredjene atribute za klasifikaciju
# features = ["Petal_Length",  "Petal_Width"]

x_original = df[features]

# Standardize attributes (subtract mean and divide with stddev)
x = pd.DataFrame(preprocessing.scale(x_original))

# Optional normalize
x = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(x))

# Add column names to vector X (because we create a pandas dataframe)
x.columns = features

# Extract target vector y
y = df["Name"]

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)

# Parameters for kNN
k = 4                   # number of neighbours to try
p = 1                   # Power for Minkowski metric

# We construct kNN classifier and perform classification
clf = KNeighborsClassifier(n_neighbors=k, p=p)
clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)

# Create confusion matrix and display it
cnf_matrix = met.confusion_matrix(y_test, y_predicted)
print("Confusion matrix", cnf_matrix, sep="\n")
print("\n")

# Get accuracy metric
accuracy = met.accuracy_score(y_test, y_predicted)
print("Accuracy", accuracy)
print("\n")

# Get other classification metrics
class_report = met.classification_report(y_test, y_predicted)
print("Classification report", class_report, sep="\n")
