# %%
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
# from utils import *
import argparse
class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 128),           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block4 = nn.Sequential(
            nn.Linear(in_features = 256*4*4, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 10)
        )
        self.flatten = nn.Flatten()
        self.softmax = F.softmax
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        logits = self.block4(x)
        pred = self.softmax(logits, dim=1)

        return logits,pred
        
def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=int, default=1,
        help="flag to decide whether to train")
    parser.add_argument("--epochs", type=int, default=50,
        help="training epochs")
    parser.add_argument("--seed", type=int, default=3312,
        help="seed of the experiment")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--batch_size", type=int, default=64,
        help="the batch size training samples and test samples")
    args = parser.parse_args()
    return args

# %%
args = parse_args()
if not os.path.exists("./output"):
    os.mkdir("./output")
if not os.path.exists("./checkpoints"):
    os.mkdir("./checkpoints")
version = "v1"
output_root = f"./output/{version}"
if not os.path.exists(output_root):
    os.mkdir(output_root)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(32),
    transforms.ToTensor()
])


model = NN(num_classes=10)
model.load_state_dict(torch.load(f"./checkpoints/{version}.pt"))
model.eval()

test_dataset = datasets.FashionMNIST(root='./dataset', train=False,
                                    transform=trans, download=True)
test_loader_no_shuffle = DataLoader(dataset=test_dataset,
                        batch_size=args.batch_size, shuffle=False)
conv_features = []
relu_features = []
final_features = []
def conv_hook_forward(module, fea_in, fea_out):
    conv_features.append(nn.Flatten()(fea_out))
    return None
def relu_hook_forward(module, fea_in, fea_out):
    relu_features.append(nn.Flatten()(fea_out))
    return None

h_conv = model.block3[6].register_forward_hook(hook=conv_hook_forward)
h_relu = model.block3[8].register_forward_hook(hook=relu_hook_forward)

with torch.no_grad():
    for x,y in test_loader_no_shuffle:
        logits, pred = model.forward(x)
        final_features.append(logits)

conv_features = torch.cat(conv_features, dim = 0).numpy()
relu_features = torch.cat(relu_features, dim = 0).numpy()
final_features = torch.cat(final_features, dim = 0).numpy()


# %%

def get_original_pairwise_affinities(X:np.array([]), 
                                     perplexity=30):

    '''
    Function to obtain affinities matrix.
    '''

    n = len(X)

    print("Computing Pairwise Affinities....")

    p_ij = np.zeros(shape=(n,n))
    for i in range(0,n):
        
        # Equation 1 numerator
        diff = X[i]-X
        sigma_i = grid_search(diff, i, perplexity) # Grid Search for σ_i
        norm = np.linalg.norm(diff, axis=1)
        p_ij[i,:] = np.exp(-norm**2/(2*sigma_i**2))

        # Set p = 0 when j = i
        np.fill_diagonal(p_ij, 0)
        
        # Equation 1 
        p_ij[i,:] = p_ij[i,:]/np.sum(p_ij[i,:])

    # Set 0 values to minimum numpy value (ε approx. = 0) 
    epsilon = np.nextafter(0,1)
    p_ij = np.maximum(p_ij,epsilon)

    print("Completed Pairwise Affinities Matrix. \n")

    return p_ij

def grid_search(diff_i, i, perplexity):

    '''
    Helper function to obtain sigma's based on user-specified perplexity.
    '''

    result = np.inf # Set first result to be infinity

    norm = np.linalg.norm(diff_i, axis=1)
    std_norm = np.std(norm) # Use standard deviation of norms to define search space

    sigma = std_norm
    for sigma_search in np.linspace(0.01*std_norm,5*std_norm,200):

        # Equation 1 Numerator
        p = np.exp(-norm**2/(2*sigma_search**2)) 

        # Set p = 0 when i = j
        p[i] = 0 

        # Equation 1 (ε -> 0) 
        epsilon = np.nextafter(0,1)
        p_new = np.maximum(p/np.sum(p),epsilon)
        
        # Shannon Entropy
        H = -np.sum(p_new*np.log2(p_new))
        
        # Get log(perplexity equation) as close to equality
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sigma = sigma_search
    
    return sigma

def get_symmetric_p_ij(p_ij:np.array([])):

    '''
    Function to obtain symmetric affinities matrix utilized in t-SNE.
    '''
        
    print("Computing Symmetric p_ij matrix....")

    n = len(p_ij)
    p_ij_symmetric = np.zeros(shape=(n,n))
    for i in range(0,n):
        for j in range(0,n):
            p_ij_symmetric[i,j] = (p_ij[i,j] + p_ij[j,i]) / (2*n)
    
    # Set 0 values to minimum numpy value (ε approx. = 0)
    epsilon = np.nextafter(0,1)
    p_ij_symmetric = np.maximum(p_ij_symmetric,epsilon)

    print("Completed Symmetric p_ij Matrix. \n")

    return p_ij_symmetric

def initialization(X: np.array([]),
                   n_dimensions = 2):

    return np.random.normal(loc=0,scale=1e-4,size=(len(X),n_dimensions))

def get_low_dimensional_affinities(Y:np.array([])):
    '''
    Obtain low-dimensional affinities.
    '''

    n = len(Y)
    q_ij = np.zeros(shape=(n,n))

    for i in range(0,n):

        # Equation 4 Numerator
        diff = Y[i]-Y
        norm = np.linalg.norm(diff, axis=1)
        q_ij[i,:] = (1+norm**2)**(-1)

    # Set p = 0 when j = i
    np.fill_diagonal(q_ij, 0)

    # Equation 4 
    q_ij = q_ij/q_ij.sum()

    # Set 0 values to minimum numpy value (ε approx. = 0)
    epsilon = np.nextafter(0,1)
    q_ij = np.maximum(q_ij,epsilon)

    return q_ij
def get_gradient(p_ij: np.array([]),
                q_ij: np.array([]),
                Y: np.array([])):
    '''
    Obtain gradient of cost function at current point Y.
    '''

    n = len(p_ij)

    # Compute gradient
    gradient = np.zeros(shape=(n, Y.shape[1]))
    for i in range(0,n):

        # Equation 5
        diff = Y[i]-Y
        A = np.array([(p_ij[i,:] - q_ij[i,:])])
        B = np.array([(1+np.linalg.norm(diff,axis=1))**(-1)])
        C = diff
        gradient[i] = 4 * np.sum((A * B).T * C, axis=0)

    return gradient  

def tSNE(X: np.array([]), 
        perplexity = 30,
        T = 1000, 
        eta = 200,
        early_exaggeration = 12,
        n_dimensions = 2):
    
    n = len(X)

    # Get original affinities matrix 
    p_ij = get_original_pairwise_affinities(X, perplexity)
    p_ij_symmetric = get_symmetric_p_ij(p_ij)
    
    # Initialization
    Y = np.zeros(shape=(T, n, n_dimensions))
    Y_minus1 = np.zeros(shape=(n, n_dimensions))
    Y[0] = Y_minus1
    Y1 = initialization(X, n_dimensions)
    Y[1] = np.array(Y1)

    print("Optimizing Low Dimensional Embedding....")
    # Optimization
    for t in range(1, T-1):
        
        # Momentum & Early Exaggeration
        if t < 250:
            alpha = 0.5
            early_exaggeration = early_exaggeration
        else:
            alpha = 0.8
            early_exaggeration = 1

        # Get Low Dimensional Affinities
        q_ij = get_low_dimensional_affinities(Y[t])

        # Get Gradient of Cost Function
        gradient = get_gradient(early_exaggeration*p_ij_symmetric, q_ij, Y[t])

        # Update Rule
        Y[t+1] = Y[t] - eta * gradient + alpha * (Y[t] - Y[t-1]) # Use negative gradient 

        # Compute current value of cost function
        if t % 50 == 0 or t == 1:
            cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
            print(f"Iteration {t}: Value of Cost Function is {cost}")

    print(f"Completed Embedding: Final Value of Cost Function is {np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))}")
    solution = Y[-1]

    return solution, Y

# %%
Y,_ = tSNE(conv_features)



