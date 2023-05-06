import numpy as np
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as tsne

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
    plt.legend()
    if ttl:
        plt.title(ttl)
    if savepth:
        plt.savefig(savepth, bbox_inches="tight")

def featureVisualizationPCA(dataset, features, label_map):
    pass

class TSNE():
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.model = tsne(n_components=self.n_components)
    
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

# class PCA_myself():
#     def __init__(self, n_components = 2):
#         self.n_components = n_components
#         self.model = PCA(n_components=self.n_components)
    
#     def fit(self, features):
#         self.pca_results = self.model.fit_transform(features)
    
#     def visualization(self, labels, n_classes = 10, savepth=None):
#         plt.clf()
#         for cl in range(n_classes):
#             indices = np.where(labels==cl)
#             indices = indices[0]
#             # print(len(indices))
#             plt.scatter(self.pca_results[indices,0], self.pca_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
#         plt.legend()
#         plt.title("PCA")
#         if savepth:
#             plt.savefig(savepth, bbox_inches="tight")


class PCA():
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.pca_results = None

    def fit(self, x):
        self.mean = np.mean(x, axis = 0)
        x = x-self.mean
        cov = np.cov(x.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]
        self.pca_results = np.dot(x, self.components.T)

    # def transform(self, x):
    #     x = x-self.mean
    #     return np.dot(x, self.components.T)
    
    # def fit_transform(self, x):
    #     self.fit(x)
    #     return self.transform(x)
    
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

# class TSNE():
#     def __init__(self, n_components = 2, perplexity = 30, maxiter = 1000):
#         self.n_components = n_components
#         self.perplexity = perplexity
#         self.maxiter = maxiter
#         self.tsne_result = None
    
#     def Hbeta(self, D, beta):
#         P = np.exp(-D * beta)
#         sumP = np.sum(P)
#         H = np.log(sumP) + beta * np.sum(D * P) / sumP
#         P = P / sumP
#         return H, P

#     def x2p(self, X, perplexity):
#         # Compute pairwise distances
#         sum_X = np.sum(np.square(X), axis=1)
#         D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

#         # Initialize some variables
#         P = np.zeros_like(D)
#         beta = np.ones_like(D)
#         logU = np.log(perplexity)

#         # Iterate over all datapoints
#         for i in range(X.shape[0]):
#             # Compute the Gaussian kernel and entropy for the current precision
#             betamin = -np.inf
#             betamax = np.inf
#             Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))]
#             H, thisP = self.Hbeta(Di, beta[i])

#             # Evaluate whether the perplexity is within tolerance
#             Hdiff = H - logU
#             tries = 0
#             while np.abs(Hdiff) > 1e-5 and tries < 50:
#                 # If not, increase or decrease precision
#                 if Hdiff > 0:
#                     betamin = beta[i].copy()
#                     if betamax == np.inf or betamax == -np.inf:
#                         beta[i] = beta[i] * 2.
#                     else:
#                         beta[i] = (beta[i] + betamax) / 2.
#                 else:
#                     betamax = beta[i].copy()
#                     if betamin == np.inf or betamin == -np.inf:
#                         beta[i] = beta[i] / 2.
#                     else:
#                         beta[i] = (beta[i] + betamin) / 2.

#                 # Recompute the entropy and Gaussian kernel with the new precision
#                 H, thisP = self.Hbeta(Di, beta[i])
#                 Hdiff = H - logU
#                 tries += 1

#             # Set the final row of P
#             P[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))] = thisP

#         # Return final P-matrix
#         return P

#     def fit(self, X):
#         # Initialize variables
#         X = X.astype('float64')
#         Y = np.random.randn(X.shape[0], self.n_components)
#         P = self.x2p(X, self.perplexity)
#         P = P + np.transpose(P)
#         P = P / np.sum(P)
#         prev_cost = 0
#         # Run iterations
#         for i in range(self.max_iter):
#             # Compute pairwise affinities
#             sum_Y = np.sum(np.square(Y), axis=1)
#             num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
#             num[range(X.shape[0]), range(X.shape[0])] = 0
#             Q = num / np.sum(num)
#             Q = np.maximum(Q, 1e-12)

#             # Compute gradient
#             PQ = P - Q
#             for j in range(X.shape[0]):
#                 Y[j, :] = Y[j, :] + np.sum(np.tile(PQ[:, j] * num[:, j], (no_dims, 1)).T * (Y[j, :] - Y), axis=0)

#             # Compute cost and check for convergence
#             cost = np.sum(P * np.log(P / Q))
#             if i > 0 and np.abs(cost - prev_cost) < 1e-5:
#                 break
#             prev_cost = cost

#         # Return final embedding
#         self.tsne_result = Y
