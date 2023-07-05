# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
print(dataset.head())
acc = []

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = set(stopwords.words('english')) - set(['not','ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                                                    "doesn't", 'hadn', "hadn't",'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                                                    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',"needn't", 'shan',
                                                    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                                                    "won't",'wouldn', "wouldn't",'don', "don't", 'no', 'nor'])
##  all_stopwords.remove
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(all_stopwords)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = len(dataset)*3)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 0)
lg.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_lg = lg.predict(X_test)
cm_lg = confusion_matrix(y_test, y_pred_lg)
print(cm_lg)
print("Logistic Regression accuracy \n",accuracy_score(y_test, y_pred_lg))
acc.append(accuracy_score(y_test, y_pred_lg))

# Decision Tree Classification

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_dt = dt.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)
print("Decision Tree accuracy \n",accuracy_score(y_test, y_pred_dt))
acc.append(accuracy_score(y_test, y_pred_dt))

# K-Nearest Neighbors (K-NN)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
print("KNN accuracy \n",accuracy_score(y_test, y_pred_knn))
acc.append(accuracy_score(y_test, y_pred_knn))

# Kernel SVM

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
kernel_svc = SVC(kernel = 'rbf', random_state = 0)
kernel_svc.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_kernel = kernel_svc.predict(X_test)
cm_kernel = confusion_matrix(y_test, y_pred_kernel)
print(cm_kernel)
print("Kernel SVM accuracy \n",accuracy_score(y_test, y_pred_kernel))
acc.append(accuracy_score(y_test, y_pred_kernel))

# Naive Bayes

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_nb = nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(cm_nb)
print("Naive Bayes accuracy \n",accuracy_score(y_test, y_pred_nb))
acc.append(accuracy_score(y_test, y_pred_nb))

# Random Forest Classification

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 27, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_rf = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)
print("Random Forest accuracy \n",accuracy_score(y_test, y_pred_rf))
acc.append(accuracy_score(y_test, y_pred_rf))

# Support Vector Machine (SVM)

# Training the SVM model on the Training set
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 0)
svc.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(cm_svc)
print("SVM accuracy \n",accuracy_score(y_test, y_pred_svc))
acc.append(accuracy_score(y_test, y_pred_svc))
acc = [round(j * 100,2) for j in acc]

fig = plt.figure(figsize = (10, 6))

# creating the accuracy bar plot
models = ["Logistic","Decision Tree","KNN","Kernel SVM","Naive Bayes","Random Forrest","SVM"]
plt.bar(models, acc,width = 0.5)

for i, v in enumerate(acc):
    plt.text(i, v + 0.01, str(v)+"%",horizontalalignment="center",verticalalignment="bottom")
plt.xlabel("Models")
plt.ylabel("Accuracy %")
plt.title("Model Accuracy")
plt.savefig("image/accuracy.png")
plt.show()
