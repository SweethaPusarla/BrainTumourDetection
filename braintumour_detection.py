import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths for data directories
training_path = r'C:\Users\sweet\Downloads\braintumour\archive\Training'
testing_path = r'C:\Users\sweet\Downloads\braintumour\archive\Testing'

# Classes
classes = {'no_tumor': 0, 'pituitary_tumor': 1}

X = []
Y = []

# Load and preprocess training images
for cls, label in classes.items():
    cls_path = os.path.join(training_path, cls)
    for filename in os.listdir(cls_path):
        img_path = os.path.join(cls_path, filename)
        img = cv2.imread(img_path, 0)
        if img is not None:
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

# Print dataset info
print(np.unique(Y))
print(pd.Series(Y).value_counts())
print(X.shape, X_updated.shape)

# Display sample image
plt.imshow(X[0], cmap='gray')
plt.show()

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)
print(xtrain.shape, xtest.shape)

# Normalize the data
xtrain = xtrain / 255
xtest = xtest / 255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

# PCA
pca = PCA(0.98)
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

print("PCA Training shape:", xtrain_pca.shape)
print("PCA Testing shape:", xtest_pca.shape)

# Train classifiers
lg = LogisticRegression(C=0.1)
lg.fit(xtrain_pca, ytrain)

sv = SVC()
sv.fit(xtrain_pca, ytrain)

# Print training and testing scores
print("Logistic Regression Training Score:", lg.score(xtrain_pca, ytrain))
print("Logistic Regression Testing Score:", lg.score(xtest_pca, ytest))

print("SVC Training Score:", sv.score(xtrain_pca, ytrain))
print("SVC Testing Score:", sv.score(xtest_pca, ytest))

# Predict and visualize misclassified samples
pred = sv.predict(xtest_pca)
misclassified = np.where(ytest != pred)
print("Misclassified samples:", misclassified)

# Define class labels for prediction
dec = {0: 'No Tumor', 1: 'Positive Tumor'}

# Visualize predictions
plt.figure(figsize=(12, 8))
c = 1
for i in os.listdir(os.path.join(testing_path, 'no_tumor'))[:9]:
    img_path = os.path.join(testing_path, 'no_tumor', i)
    img = cv2.imread(img_path, 0)
    if img is not None:
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1) / 255
        img1_pca = pca.transform(img1)  # Apply PCA to new image
        p = sv.predict(img1_pca)
        plt.subplot(3, 3, c)
        plt.title(dec[p[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        c += 1

plt.figure(figsize=(12, 8))
c = 1
for i in os.listdir(os.path.join(testing_path, 'pituitary_tumor'))[:16]:
    img_path = os.path.join(testing_path, 'pituitary_tumor', i)
    img = cv2.imread(img_path, 0)
    if img is not None:
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1) / 255
        img1_pca = pca.transform(img1)  # Apply PCA to new image
        p = sv.predict(img1_pca)
        plt.subplot(4, 4, c)
        plt.title(dec[p[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        c += 1

plt.show()

# Save the trained model using pickle
model_filename = 'brain_tumor_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(sv, file)  
