import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

###
# Student Name: Sharon Maldonado
# Course: CAP-4612
# Project: Naive Bayes Algorithm to Predict Lung Cancer amongst patients
###

# load the dataset
data = pd.read_csv("survey_lung_cancer.csv")

# categorical values to binary
data = pd.get_dummies(data,
                      columns=["GENDER", "YELLOW_FINGERS", "ANXIETY", "CHRONIC DISEASE", "ALLERGY", "ALCOHOL CONSUMING",
                               "COUGHING", "SWALLOWING DIFFICULTY", "CHEST PAIN"])

#  change 'no' to 0 and 'yes' to 1 in the LUNG_CANCER column
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# split data into labels and features
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# split the data into training and testing sets then implement Naive Bayes Algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# make sample prediction on test above
y_p = nb_model.predict(X_test)

# initialize accuracy, precision, recall, and f1-score to evaluate results
accuracy = accuracy_score(y_test, y_p)
precision = precision_score(y_test, y_p)
recall = recall_score(y_test, y_p)
f1 = f1_score(y_test, y_p)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
