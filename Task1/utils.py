import numpy as np
from matplotlib import pyplot as plt
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
    
    def visualization(self, labels, n_classes = 10, savepth=None):
        plt.clf()
        for cl in range(n_classes):
            indices = np.where(labels==cl)
            indices = indices[0]
            plt.scatter(self.pca_results[indices,0], self.pca_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
        plt.legend()
        plt.title("PCA")
        if savepth:
            plt.savefig(savepth, bbox_inches="tight")
            
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
                # cost = np.sum(p_ij * np.log(p_ij / q_ij))

            self.tsne_results = Y[-1]
        
    def visualization(self, labels, n_classes = 10, savepth=None):
        plt.clf()
        for cl in range(n_classes):
            indices = np.where(labels==cl)
            indices = indices[0]
            plt.scatter(self.tsne_results[indices,0], self.tsne_results[indices, 1], label=cl, alpha = 0.5, linewidth=0.5)
        plt.legend()
        plt.title("t-SNE")
        if savepth:
            plt.savefig(savepth, bbox_inches="tight")



