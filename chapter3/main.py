import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import decisionregions as dr
import logisticregressiongd as lgd
import fullbatchgradientdescent as fgd


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels: ', np.unique(y))

# Perform the train_test_split function from scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardize the features using the StandardScaler class
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training the perceptron via the scikit-learn
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# Make prediction via the predict method
y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())

# outcome of misclassified of 1 out of 45 flower examples

# Calculate the classification accuracy of the perceptron on the test dataset follows:
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# Compute classifier's prediction accuracy by combining the `predict` call with `accuracy_score`
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# Training a perceptron model using standardized training data:
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

dr.plot_decision_regions(X=X_combined_std, y=y_combined,
                         classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# # Using numpy reshape
# print(lr.predict(X_test_std[0, :].reshape(1, -1)))
#
# Execute sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)

plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

# Loss of classifying a single training
def loss_1(z):
    return - np.log(sigmoid(z))

def loss_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)

c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w, b) if y=1')

c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Implement the logistic regresion using full batch
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = fgd.LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

dr.plot_decision_regions(X=X_train_01_subset,
                         y=y_train_01_subset,
                         classifier=lrgd)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# sklearn linear_model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

dr.plot_decision_regions(X_combined_std, y_combined,
                         classifier=lr, test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Perform the prediction
print(lr.predict_proba(X_test_std[:3, :]))

# Predict using the sum
print(lr.predict_proba(X_test_std[:3, :]).sum(axis=1))


# Predict using argmax
print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))

# Predict
print(lr.predict(X_test_std[:3, :]))

print(lr.predict(X_test_std[0, :].reshape(1, -1)))



# X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
# y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
#
# lrgd = lgd.LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
# lrgd.fit(X_train_01_subset,
#          y_train_01_subset)
#
# dr.plot_decision_regions(X=X_train_01_subset,
#                          y=y_train_01_subset,
#                          classifier=lrgd)
#
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

