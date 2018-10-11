import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm, metrics
from sklearn import model_selection

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

X,y = datasets.make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=3.5, center_box=(-10.0, 10.0))

colors = np.array([x for x in "bgrcmyk"])
plt.scatter(X[:,0], X[:, 1], color=colors[y].tolist(), s=10)
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, test_size=0.3)

C=100
i=1


gamma_range = np.logspace(-3, 3, 4, endpoint=False)
for gamma in gamma_range:
    rbfsvm = svm.SVC(C=C, gamma=gamma)
    print(gamma)
    rbfsvm.fit(X_train, y_train)

    ypred = rbfsvm.predict(X_test)
    err_test = metrics.accuracy_score (ypred, y_test)
    print("test set accuracy : %.3f" % err_test)
    print(metrics.confusion_matrix(y_test, ypred))

    ypred = rbfsvm.predict(X_train)
    err_train = metrics.accuracy_score (ypred, y_train)
    print("train set accuracy : %.3f" % err_train)
    print(metrics.confusion_matrix(y_train, ypred))

    plt.subplot(4,2,i)
    i+=1
    # Créer un mesh
    h = .05 # Espacement du mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Y = rbfsvm.predict(np.c_[xx.ravel(), yy.ravel()])
    # Afficher
    Y = Y.reshape(xx.shape)
    plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], cmap=plt.cm.Paired, color=colors[y_train].tolist())
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("train set accuracy : %.3f, gamma: %.3f" % (err_train, gamma))


    plt.subplot(4,2,i)
    i+=1
    # Créer un mesh
    h = .05 # Espacement du mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Y = rbfsvm.predict(np.c_[xx.ravel(), yy.ravel()])
    # Afficher
    Y = Y.reshape(xx.shape)
    plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], cmap=plt.cm.Paired, color=colors[y_test].tolist())
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("test set accuracy : %.3f, gamma: %.3f" % (err_test, gamma))

    print("nb de vecteurs par classe: ",rbfsvm.n_support_)

plt.show()
