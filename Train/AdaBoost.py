import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv('./data.csv')

# split dataset into features and target variable
X = df.iloc[:, 9:14]
y = df.iloc[:, 14]

# split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# initialize decision tree classifier
dtc = DecisionTreeClassifier(max_depth=1)

# initialize AdaBoost classifier
abc = AdaBoostClassifier(base_estimator=dtc, n_estimators=50, learning_rate=1)

# fit AdaBoost classifier to training set
abc.fit(X_train, y_train)

# predict on testing set
y_pred = abc.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score:', accuracy)

# Gradient Boosting Machine (GBM)
from sklearn.ensemble import GradientBoostingClassifier

# initialize Gradient Boosting classifier
gbm = GradientBoostingClassifier(n_estimators=50, learning_rate=1, max_depth=1, random_state=0)

# fit GBM to training set
gbm.fit(X_train, y_train)

# predict on testing set
y_pred = gbm.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score:', accuracy)