import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_curve(y, ttl=None, savepth=None):
    plt.clf()
    plt.plot(y, "o-", linewidth=1)
    plt.xlabel("epoch")
    if ttl:
        plt.title(ttl)
    if savepth:
        plt.savefig(savepth)
def plot_curveII(y1, y2, ttl=None, savepth=None):
    plt.clf()
    plt.plot(y1, "o-", linewidth=1, label="train")
    plt.plot(y2, "o-", linewidth=1, label="test")
    plt.xlabel("epoch")
    plt.legend(loc="upper right")
    if ttl:
        plt.title(ttl)
    if savepth:
        plt.savefig(savepth, bbox_inches="tight")

def featureVisualizationPCA(dataset, features, label_map):
    pass

class TSNE_myself():
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.model = TSNE(n_components=self.n_components)
    
    def fit(self, features):
        self.tsne_results = self.model.fit_transform(features)
    
    def visualization(self, labels, n_classes = 10, savepth=None):
        plt.clf()
        for cl in range(n_classes):
            indices = np.where(labels==cl)
            indices = indices[0]
            # print(len(indices))
            plt.scatter(self.tsne_results[indices,0], self.tsne_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
        plt.legend()
        plt.title("t-SNE")
        if savepth:
            plt.savefig(savepth, bbox_inches="tight")

class PCA_myself():
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.model = PCA(n_components=self.n_components)
    
    def fit(self, features):
        self.pca_results = self.model.fit_transform(features)
    
    def visualization(self, labels, n_classes = 10, savepth=None):
        plt.clf()
        for cl in range(n_classes):
            indices = np.where(labels==cl)
            indices = indices[0]
            # print(len(indices))
            plt.scatter(self.pca_results[indices,0], self.pca_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
        plt.legend()
        plt.title("PCA")
        if savepth:
            plt.savefig(savepth, bbox_inches="tight")