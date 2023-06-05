import numpy as np
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn import metrics
import copy

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


# class TSNE():
#     def __init__(self, n_components = 2):
#         self.n_components = n_components
#         self.model = tsne(n_components=self.n_components)
    
#     def fit(self, features):
#         self.tsne_results = self.model.fit_transform(features)
    
#     def visualization(self, labels, n_classes = 10, savepth=None):
#         plt.clf()
#         for cl in range(n_classes):
#             indices = np.where(labels==cl)
#             indices = indices[0]
#             # print(len(indices))
#             plt.scatter(self.tsne_results[indices,0], self.tsne_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
#         plt.legend()
#         plt.title("t-SNE")
#         if savepth:
#             plt.savefig(savepth, bbox_inches="tight")


# NOTE: v4
# class TSNE():
#     def __init__(self, perplexity = 30, max_iter = 1000, eta = 200, early_exaggeration = 12, n_components = 2):
#         self.perplexity = perplexity
#         self.max_iter = max_iter
#         self.eta = eta
#         self.early_exaggeration = early_exaggeration
#         self.n_components = n_components

#     def grid_search(self, diff_i, i):

#         '''
#         Helper function to obtain sigma's based on user-specified perplexity.
#         '''

#         result = np.inf # Set first result to be infinity

#         norm = np.linalg.norm(diff_i, axis=1)
#         std_norm = np.std(norm)

#         sigma = std_norm
#         for sigma_search in np.linspace(0.01*std_norm, 5*std_norm, 200):
#             # calculate p_{j|i} for each j
#             p = np.exp(-norm*norm / (2*sigma_search*sigma_search))
#             p[i] = 0
#             # print(p)
#             # Set 0 values to minimum numpy value ($\epsilon$ approx. = 0) 
#             epsilon = np.finfo(p.dtype).eps
#             # NOTE: add by myself
#             p = np.maximum(p, epsilon)
#             p_new = p/np.sum(p)

#             # Shannon Entropy
#             H = -np.sum(p_new * np.log2(p_new))

#             # Determine whether $perplexity$ and $2^{H(p)}$ are enough close. 
#             if np.abs(np.log(self.perplexity)-H*np.log(2)) < result:
#                 result = np.abs(np.log(self.perplexity)-H*np.log(2))
#                 sigma = sigma_search
#         return sigma


#     def get_original_pairwise_affinities(self, X: np.array([])):
#         '''
#         Function to obtain affinities matrix p_{j|i}.
#         '''

#         m, _ = X.shape


#         p_j_given_i = np.zeros((m, m))
        
#         for i in range(m):
#             diff = X[i, :] - X
#             sigma_i = self.grid_search(diff, i)
#             norm = np.linalg.norm(diff, axis=1)
#             p_j_given_i[i, :] = np.exp(-norm*norm / (2*sigma_i*sigma_i))

#             # Set p = 0 for i == j
#             np.fill_diagonal(p_j_given_i, 0)


#         # Set 0 values to minimum numpy value ($\epsilon$ approx. = 0) 
#         epsilon = np.finfo(p_j_given_i.dtype).eps
#         p_j_given_i = np.maximum(p_j_given_i, epsilon)
#         p_j_given_i = p_j_given_i / p_j_given_i.sum(axis=1)

#         return p_j_given_i


#     def get_symmetric_p_ij(self, p_j_given_i:np.array([])):

#         '''
#         Function to obtain symmetric affinities matrix utilized in t-SNE.
#         '''
            

#         m, _ = p_j_given_i.shape
#         p_ij = (p_j_given_i + p_j_given_i.transpose()) / (2*m)

#         # Set 0 values to minimum numpy value ($\epsilon$ approx. = 0) 
#         epsilon = np.finfo(p_ij.dtype).eps
#         p_ij = np.maximum(p_ij, epsilon)


#         return p_ij

#     def initialization(self, X: np.array([])):

#         return np.random.normal(loc=0,scale=1e-4,size=(X.shape[0],self.n_components))

#     def get_low_dimensional_affinities(self, Y:np.array([])):
#         '''
#         Obtain low-dimensional affinities.
#         '''

#         m, _ = Y.shape
#         q_ij = np.zeros((m, m))

#         for i in range(m):
#             diff = Y[i, :] - Y
#             norm = np.linalg.norm(diff, axis=1)
#             q_ij[i] = np.power(1+norm*norm, -1)

#         np.fill_diagonal(q_ij, 0)

#         # Set 0 values to minimum numpy value ($\epsilon$ approx. = 0) 
#         epsilon = np.finfo(q_ij.dtype).eps
#         q_ij = np.maximum(q_ij, epsilon)
#         q_ij = q_ij / q_ij.sum()

#         return q_ij

#     def get_gradients(self, p_ij: np.array([]), q_ij: np.array([]), Y: np.array([])):
#         '''
#         Obtain gradient of cost function at current point Y.
#         '''
#         m, _ = p_ij.shape
#         A = p_ij - q_ij # shape: (m, m)

#         diff = np.zeros((m, m, Y.shape[1]))
#         for i in range(m):
#             diff[i,:] = Y[i,:] - Y
        
#         B = np.power(1+np.linalg.norm(diff, axis=2), -1) # shape: (m, m)
#         gradients = np.sum(4*(A*B)[:,:,np.newaxis]*diff, axis=1)
#         return gradients

#     def fit(self, X: np.array([])):
#         m, _ = X.shape

#         p_j_given_i = self.get_original_pairwise_affinities(X)
#         p_ij = self.get_symmetric_p_ij(p_j_given_i)

#         # Initialization
#         Y = np.zeros((self.max_iter, m, self.n_components))
#         Y[0] = np.zeros((m, self.n_components))
#         Y[1] = self.initialization(X)


#         for t in range(1, self.max_iter-1):

#             # Momentum & Early Exaggeration
#             if t < 250:
#                 alpha = 0.5
#                 early_exaggeration = self.early_exaggeration
#             else:
#                 alpha = 0.8
#                 early_exaggeration = 1
            
#             # Get Low Dimensional Affinities
#             q_ij = self.get_low_dimensional_affinities(Y[t])

#             # Get Gradient of Cost Function
#             gradients = self.get_gradients(early_exaggeration*p_ij, q_ij, Y[t])

#             # Update Rule
#             Y[t+1] = Y[t] - self.eta * gradients + alpha * (Y[t] - Y[t-1])

#             # Compute current value of cost function
#             # if t % 50 == 0 or t == 1:
#             cost = np.sum(p_ij * np.log(p_ij / q_ij))
#             print(f"Iteration {t}: Value of Cost Function is {cost}")

#         self.tsne_results = Y[-1]
#     def visualization(self, labels, n_classes = 10, savepth=None):
#         plt.clf()
#         for cl in range(n_classes):
#             indices = np.where(labels==cl)
#             indices = indices[0]
#             # print(len(indices))
#             plt.scatter(self.tsne_results[indices,0], self.tsne_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
#         plt.legend()
#         plt.title("t-SNE")
#         if savepth:
#             plt.savefig(savepth, bbox_inches="tight")

 # v5
class tSNE():
    def __init__(self, perplexity = 30, max_iter = 1000, eta = 200, early_exaggeration = 12, n_components = 2):
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.eta = eta
        self.early_exaggeration = early_exaggeration
        self.n_components = n_components
        self.model = TSNE(perplexity = perplexity,n_iter = max_iter, early_exaggeration = early_exaggeration, n_components = n_components) # If choose 'fast', the model will directly use the `sklearn.manifold.TSNE`
    def calc_entropy(self, D: np.ndarray, beta: float):
        """Calculate entropy for distance and betavalue (1/(2\sigma_i^2))

        Args:
            D (np.ndarray): distance between i and j datapoint
            beta (float): a variable for (1 / (2\sigma_i^2))

        Returns:
            _type_: entropy for this 
        """
        P=np.exp(-D*beta)
        sumP=sum(P)
        sumP=np.maximum(sumP,1e-200)
        H=np.log(sumP) + beta * np.sum(D * P) / sumP
        return H

    def calc_matrix_P(self, X: np.ndarray):
        """Calculate matrix P for input matrix X

        Args:
            X (np.ndarray): input data points; shape: (N, d)

        Returns:
            _type_: _description_
        """
        entropy=np.log(self.perplexity)
        n1,n2=X.shape
        D=np.square(metrics.pairwise_distances(X))
        D_sort=np.argsort(D,axis=1)
        P=np.zeros((n1,n1))
        for i in range(n1):
            Di=D[i,D_sort[i,1:]]
            P[i,D_sort[i,1:]]=self.calc_p(Di,entropy=entropy)
        P=(P+np.transpose(P))/(2*n1)
        P=np.maximum(P,1e-100)
        return P
    def calc_p(self, D: np.ndarray, entropy: float, iter_times=50):
        beta=1.0
        H=self.calc_entropy(D,beta)
        error=H-entropy
        k=0
        betamin=-np.inf
        betamax=np.inf
        while np.abs(error)>1e-4 and k<=iter_times:
            if error > 0:
                betamin=copy.deepcopy(beta)
                if betamax==np.inf:
                    beta=beta*2
                else:
                    beta=(beta+betamax)/2
            else:
                betamax=copy.deepcopy(beta)
                if betamin==-np.inf:
                    beta=beta/2
                else:
                    beta=(beta+betamin)/2
            H=self.calc_entropy(D,beta)
            error=H-entropy
            k+=1
        P=np.exp(-D*beta)
        P=P/np.sum(P)
        return P
    
    def initialization(self, X: np.array([])):

        return np.random.normal(loc=0,scale=1e-4,size=(X.shape[0],self.n_components))

    def get_low_dimensional_affinities(self, Y:np.array([])):
        '''
        Obtain low-dimensional affinities.
        '''

        m, _ = Y.shape
        q_ij = np.zeros((m, m))

        for i in range(m):
            diff = Y[i, :] - Y
            norm = np.linalg.norm(diff, axis=1)
            q_ij[i] = np.power(1+norm*norm, -1)

        np.fill_diagonal(q_ij, 0)

        # Set 0 values to minimum numpy value ($\epsilon$ approx. = 0) 
        epsilon = np.finfo(q_ij.dtype).eps
        q_ij = np.maximum(q_ij, epsilon)
        q_ij = q_ij / q_ij.sum()

        return q_ij

    def get_gradients(self, p_ij: np.array([]), q_ij: np.array([]), Y: np.array([])):
        '''
        Obtain gradient of cost function at current point Y.
        '''
        m, _ = p_ij.shape
        A = p_ij - q_ij # shape: (m, m)

        diff = np.zeros((m, m, Y.shape[1]))
        for i in range(m):
            diff[i,:] = Y[i,:] - Y
        
        B = np.power(1+np.linalg.norm(diff, axis=2), -1) # shape: (m, m)
        gradients = np.sum(4*(A*B)[:,:,np.newaxis]*diff, axis=1)
        return gradients

    def fit(self, X: np.array([]), pca_results: np.array([]), speed='fast'):
        if speed.lower() == 'fast':
            self.tsne_results = self.model.fit_transform(X)
        else:
            m, _ = X.shape
            self.eta = max(m / self.early_exaggeration / 4, 50)

            # p_j_given_i = self.get_original_pairwise_affinities(X)
            # p_ij = self.get_symmetric_p_ij(p_j_given_i)
            p_ij = self.calc_matrix_P(X)

            # Initialization
            Y = np.zeros((self.max_iter, m, self.n_components))
            Y[0] = np.zeros((m, self.n_components))
            Y[1] = pca_results


            for t in range(1, self.max_iter-1):

                # Momentum & Early Exaggeration
                if t < 250:
                    alpha = 0.5
                    early_exaggeration = self.early_exaggeration
                else:
                    alpha = 0.8
                    early_exaggeration = 1
                
                # Get Low Dimensional Affinities
                q_ij = self.get_low_dimensional_affinities(Y[t])

                # Get Gradient of Cost Function
                gradients = self.get_gradients(early_exaggeration*p_ij, q_ij, Y[t])

                # Update Rule
                Y[t+1] = Y[t] - self.eta * gradients + alpha * (Y[t] - Y[t-1])

                # Compute current value of cost function
                # if t % 50 == 0 or t == 1:
                cost = np.sum(p_ij * np.log(p_ij / q_ij))
                print(f"Iteration {t}: Value of Cost Function is {cost}")

            self.tsne_results = Y[-1]
        
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
        self.pca_results = None

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
