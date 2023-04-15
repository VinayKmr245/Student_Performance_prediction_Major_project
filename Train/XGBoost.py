import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# importing the dataset
dataset = pd.read_csv('./data.csv')
le = LabelEncoder()
X = dataset.iloc[:, 9:14].values
y=dataset.iloc[:, 14].values
y = le.fit_transform(y)

# splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# XGBoost
from xgboost import XGBClassifier
print(X_train,y_train)
classifier = XGBClassifier(objective='multi:softprob', eval_metric='merror',use_label_encoder=False)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
import seaborn as sns
sns.heatmap(cm, annot=True)

# Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
sns.heatmap(cm, annot=True)