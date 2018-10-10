import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm, metrics
from sklearn import model_selection

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

'''
iris = datasets.load_iris()
X = iris.data#[:, :2]  # we only take the first two features.
y = iris.target'''


X,y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=4.0, center_box=(-10.0, 10.0))

colors = np.array([x for x in "bgrcmyk"])
plt.scatter(X[:,0], X[:, 1], color=colors[y].tolist(), s=10)
plt.show()



'''
X2D = X[:, 2:4]

#entraîner un modèle (svm linéaire)
linsvm = svm.LinearSVC(C=10)
linsvm.fit(X2D, y)
ypred = linsvm.predict(X2D)
err_train = 1 - metrics.accuracy_score (ypred, y)
print("Train error: %.3f" % err_train)

print(metrics.confusion_matrix(y, ypred))

# Créer un mesh
h = .02 # Espacement du mesh
x_min, x_max = X2D[:, 0].min() - 1, X2D[:, 0].max() + 1
y_min, y_max = X2D[:, 1].min() - 1, X2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
Y = linsvm.predict(np.c_[xx.ravel(), yy.ravel()])
# Afficher
Y = Y.reshape(xx.shape)
plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X2D[:, 0], X2D[:, 1], cmap=plt.cm.Paired, color=colors[y].tolist())
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

#validation croisée
X_train, X_test, y_train, y_test = model_selection.train_test_split(X2D, y, train_size=0.7, test_size=0.3)

for C in np.logspace(-10, 10, 10, endpoint=False):
    print(C)
    linsvm2 = svm.LinearSVC(C=C)
    linsvm2.fit(X_train, y_train)
    ypred = linsvm2.predict(X_test)
    err_train = metrics.accuracy_score (ypred, y_test)
    print("test set:")
    print("acc error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_test, ypred))

    ypred = linsvm2.predict(X_train)
    err_train = metrics.accuracy_score (ypred, y_train)
    print("train set:")
    print("acc error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_train, ypred))

print()
print('SPACE')
print()
'''
X0D = X[:, :2]

#entraîner un modèle (svm linéaire)
linsvm = svm.LinearSVC(C=10)
linsvm.fit(X0D, y)
ypred = linsvm.predict(X0D)
err_train = metrics.accuracy_score (ypred, y)
print("accuracy: %.3f" % err_train)

print(metrics.confusion_matrix(y, ypred))


#validation croisée
X_train, X_test, y_train, y_test = model_selection.train_test_split(X0D, y, train_size=0.7, test_size=0.3)

i=1
nb_c=5
for C in np.logspace(0, 100000, 5, endpoint=False):
    plt.subplot(2, nb_c, i)
    i+=1
    linsvm2 = svm.LinearSVC(C=C)
    linsvm2.fit(X_train, y_train)
    ypred = linsvm2.predict(X_test)
    err_train = metrics.accuracy_score (ypred, y_test)
    print("test set accuracy error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_test, ypred))

    ypred = linsvm2.predict(X_train)
    err_train = metrics.accuracy_score (ypred, y_train)
    print("train set accuracy error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_train, ypred))

    # Créer un mesh
    h = .02 # Espacement du mesh
    x_min, x_max = X0D[:, 0].min() - 1, X0D[:, 0].max() + 1
    y_min, y_max = X0D[:, 1].min() - 1, X0D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Y = linsvm2.predict(np.c_[xx.ravel(), yy.ravel()])
    # Afficher
    Y = Y.reshape(xx.shape)
    plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X0D[:, 0], X0D[:, 1], cmap=plt.cm.Paired, color=colors[y].tolist())
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
plt.show()


X0D = X[:, :2]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X0D, y, train_size=0.7, test_size=0.3)


i=1
nb_g = 5
for gamma in np.linspace(0.1, 5, nb_g, endpoint=False):
    plt.subplot(2,nb_g,i)
    rbfsvm = svm.SVC(C=10, gamma=gamma)
    rbfsvm.fit(X_train, y_train)
    i+=1

    ypred = rbfsvm.predict(X_test)
    err_train = metrics.accuracy_score (ypred, y_test)
    print("test set accuracy error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_test, ypred))

    ypred = rbfsvm.predict(X_train)
    err_train = metrics.accuracy_score (ypred, y_train)
    print("train set accuracy error: %.3f" % err_train)
    print(metrics.confusion_matrix(y_train, ypred))

    # Créer un mesh
    h = .02 # Espacement du mesh
    x_min, x_max = X0D[:, 0].min() - 1, X0D[:, 0].max() + 1
    y_min, y_max = X0D[:, 1].min() - 1, X0D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Y = rbfsvm.predict(np.c_[xx.ravel(), yy.ravel()])
    # Afficher
    Y = Y.reshape(xx.shape)
    plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X0D[:, 0], X0D[:, 1], cmap=plt.cm.Paired, color=colors[y].tolist())
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    print("nb de vecteurs par classe: ",rbfsvm.n_support_)

plt.show()




i=1
nb_c = 5
nb_g = 5

for C in np.logspace(0, 10, nb_c, endpoint=False):
    for gamma in np.linspace(0.1, 5, nb_g, endpoint=False):
        plt.subplot(nb_g,nb_c,i)
        rbfsvm = svm.SVC(C=C, gamma=gamma)
        rbfsvm.fit(X_train, y_train)
        i+=1

        ypred = rbfsvm.predict(X_test)
        err_train = metrics.accuracy_score (ypred, y_test)
        print("test set accuracy error: %.3f" % err_train)
        print(metrics.confusion_matrix(y_test, ypred))

        ypred = rbfsvm.predict(X_train)
        err_train = metrics.accuracy_score (ypred, y_train)
        print("train set accuracy error: %.3f" % err_train)
        print(metrics.confusion_matrix(y_train, ypred))

        # Créer un mesh
        h = .02 # Espacement du mesh
        x_min, x_max = X0D[:, 0].min() - 1, X0D[:, 0].max() + 1
        y_min, y_max = X0D[:, 1].min() - 1, X0D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))
        Y = rbfsvm.predict(np.c_[xx.ravel(), yy.ravel()])
        # Afficher
        Y = Y.reshape(xx.shape)
        plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X0D[:, 0], X0D[:, 1], cmap=plt.cm.Paired, color=colors[y].tolist())
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        print("nb de vecteurs par classe: ",rbfsvm.n_support_)

plt.show()
