import pickle

from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


def train(X_train_valid, X_train, X_valid, X_test, y_train, y_valid, y_train_valid):
    img = exposure.equalize_adapthist(X_train_valid[4], clip_limit=0.01)

    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    #plt.imshow(X_train_valid[4].reshape(-1, 64), cmap="gray")
    plt.title("Original", fontsize=16)
    plt.subplot(122)
    #plt.imshow(img.reshape(-1, 64), cmap="gray")
    plt.title("Color histogramme", fontsize=16)
    print("Finished")

    pca = PCA()
    pca.fit(X_train)
    variance_rate = 0.99
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= variance_rate) + 1

    plt.figure(figsize=(6,4))
    plt.plot(cumsum, linewidth=3)
    plt.axis([0, 400, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, variance_rate], "k:")
    plt.plot([0, d], [variance_rate, variance_rate], "k:")
    plt.plot(d, variance_rate, "ko")
    plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
                arrowprops=dict(arrowstyle="->"), fontsize=16)
    plt.grid(True)

    pca = PCA(variance_rate)

    X_train_pca = pca.fit_transform(X_train)

    X_valid_pca = pca.transform(X_valid)

    X_test_pca = pca.transform(X_test)

    X_recovered = pca.inverse_transform(X_train_pca)

    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    #plt.imshow(X_train[1].reshape(-1, 64), cmap="gray")
    plt.title("Original", fontsize=16)
    plt.subplot(122)
    #plt.imshow(X_recovered[1].reshape(-1, 64), cmap="gray")
    plt.title("Compressed", fontsize=16)

    k_range = range(5, 150, 5)
    kmeans_per_k = []
    for k in k_range:
        print("k={}".format(k))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train_pca)
        kmeans_per_k.append(kmeans)
    
    

    silhouette_scores = [silhouette_score(X_train_pca, model.labels_)
                        for model in kmeans_per_k]
    best_index = np.argmax(silhouette_scores)
    best_k = k_range[best_index]
    best_score = silhouette_scores[best_index]

    plt.figure(figsize=(8, 3))
    plt.plot(k_range, silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.plot(best_k, best_score, "rs")
    print(best_k)
    best_model = kmeans_per_k[best_index]
    #plt.show()

    def plot_faces(faces, labels, n_cols=5):
        faces = faces.reshape(-1, 64, 64)
        n_rows = (len(faces) - 1) // n_cols + 1
        plt.figure(figsize=(n_cols, n_rows * 1.1))
        for index, (face, label) in enumerate(zip(faces, labels)):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(face, cmap="gray")
            plt.axis("off")
            plt.title(label)
        #plt.show()

    for cluster_id in np.unique(best_model.labels_):
        print("Cluster", cluster_id)
        in_cluster = best_model.labels_==cluster_id
        faces = X_train[in_cluster]
        labels = y_train[in_cluster]
        plot_faces(faces, labels)
    
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train_pca, y_train)
    clf.score(X_valid_pca, y_valid)
    X_train_reduced = best_model.transform(X_train_pca)
    X_valid_reduced = best_model.transform(X_valid_pca)
    X_test_reduced = best_model.transform(X_test_pca)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train_reduced, y_train)
        
    clf.score(X_valid_reduced, y_valid)

    X_train_reduced.shape

    for n_clusters in k_range:
        pipeline = Pipeline([
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),
            ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=42))
        ])
        pipeline.fit(X_train_pca, y_train)
        print(n_clusters, pipeline.score(X_valid_pca, y_valid))
    X_train_extended = np.c_[X_train_pca, X_train_reduced]
    X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
    X_test_extended = np.c_[X_test_pca, X_test_reduced]
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train_extended, y_train)
    clf.score(X_valid_extended, y_valid)


    X_train_valid_pca = PCA(X_train_valid)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_pca, y_train)
    model.score(X_valid_pca, y_valid)

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train_valid, y_train_valid, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    pickle.dump(model, open('../../models/rdf_nestimators_150_state_42.pkl', 'wb'))