import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# prepare data
input_dir = 'data-p2/top-box' # directory where image data is stored
categories = ['empty', 'not_empty','none']

data = []  # To store image data
labels = []  # To store corresponding labels
for category_idx, category in enumerate(categories):
    category_dir = os.path.join(input_dir, category)
    for file in os.listdir(category_dir):
        img_path = os.path.join(category_dir, file)
        img = imread(img_path)  # Read image
        img = resize(img, (30, 10))
        data.append(img.flatten())  # Flatten image and store in data
        labels.append(category_idx)  # Store corresponding label index

data = np.asarray(data)  # Convert to numpy array
labels = np.asarray(labels)  # Convert to numpy array

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)  # Predict on test data

score = accuracy_score(y_prediction, y_test)  # Calculate accuracy

print('{}% of samples were correctly classified'.format(str(score * 100)))

# Serialize and save the model
pickle.dump(best_estimator, open('eel.p', 'wb'))

# Visualize Decision Boundary using PCA
# Fit PCA on training data
pca = PCA(n_components=2)
pca.fit(x_train)

# Transform test data and meshgrid to reduced feature space
x_test_pca = pca.transform(x_test)
h = .02  # step size in the mesh
xx_pca, yy_pca = np.meshgrid(np.arange(x_test_pca[:, 0].min() - 1, x_test_pca[:, 0].max() + 1, h),
                             np.arange(x_test_pca[:, 1].min() - 1, x_test_pca[:, 1].max() + 1, h))

# Transform meshgrid to original feature space
xx_original = pca.inverse_transform(np.c_[xx_pca.ravel(), yy_pca.ravel()])
Z = best_estimator.predict(xx_original).reshape(xx_pca.shape)

# Plot decision boundary
plt.contourf(xx_pca, yy_pca, Z, alpha=0.8, cmap=plt.cm.coolwarm)
scatter = plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)

# Add legend
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

# Annotate regions
class_names = ['empty', 'not_empty','none']
for i, class_name in enumerate(class_names):
    # Find the centroid of each class region
    mask = Z == i
    if mask.any():
        centroid = np.mean(np.c_[xx_pca[mask], yy_pca[mask]], axis=0)
        plt.text(centroid[0], centroid[1], class_name, color='white', fontsize=12, ha='center')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary (PCA)')
plt.show()

# Classification Report
from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_prediction))
print("Classification Report:")
print(classification_report(y_test, y_prediction))
