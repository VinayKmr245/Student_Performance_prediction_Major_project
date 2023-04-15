import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset from the CSV file
dataset = pd.read_csv('./data.csv', header=None)
X = dataset.iloc[:, 9:14].values
y = dataset.iloc[:, 14].values

# splitting dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting SVM to dataset
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# predict test set result
y_pred = classifier.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# plot confusion matrix
class_names=[0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))