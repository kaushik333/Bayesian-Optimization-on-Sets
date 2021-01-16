# %%

import math
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

# %%

class SetKernel(Kernel):
    def __init__(self, num_set_size=1, num_feat_size=2, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):

        self.num_set_size = num_set_size
        self.num_feat_size = num_feat_size
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def is_stationary(self):
         return False

    def __call__(self, X, Y=None, eval_gradient=False):

        X = X.reshape(X.shape[0], self.num_set_size, self.num_feat_size)

        if Y is None: 

            # compute K matrix 
            K = np.zeros((X.shape[0], X.shape[0]))
            if self.anisotropic:
                K_grad = np.zeros((X.shape[0], X.shape[0], self.num_feat_size))
            else:
                K_grad = np.zeros((X.shape[0], X.shape[0], 1))

            for i in range(X.shape[0]):
                for j in range(X.shape[0]):

                    # compute kset between each element of setval1 and setval2
                    setval1 = X[i,:,:]
                    setval2 = X[j,:,:] 

                    # ij'th element of the K matrix
                    Kij = 0
                    Kij_grad = 0
                    # compute k_set
                    for m in range(setval1.shape[0]):
                        for n in range(setval2.shape[0]):

                            dist = np.sum((setval1[m,:] - setval2[n,:])**2 / self.length_scale) 
                            Kij_val = np.exp(-0.5*dist)
                            Kij += Kij_val

                            if eval_gradient:
                                if self.hyperparameter_length_scale.fixed:
                                    Kij_grad_val = 0
                                else:
                                    dist_new = np.sum((setval1[m,:] - setval2[n,:])**2 / self.length_scale**2) 
                                    Kij_grad_val = Kij_val * dist

                                Kij_grad += Kij_grad_val

                    # normalize by product of cardinality of sets
                    Kij /= setval1.shape[0]*setval2.shape[0]
                    Kij_grad /= setval1.shape[0]*setval2.shape[0]
                    K[i,j] = Kij
                    K_grad[i,j,:] = Kij_grad

            if eval_gradient:
                return K, K_grad
            else:
                return K

        else: 

            # compute K matrix 
            K = np.zeros((X.shape[0], Y.shape[0]))

            Y = Y.reshape(Y.shape[0], self.num_set_size, self.num_feat_size)
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):

                    setval1 = X[i,:,:]
                    setval2 = Y[j,:,:] 

                    # ij'th element of the K matrix
                    Kij = 0
                    
                    # compute k_set
                    for m in range(setval1.shape[0]):
                        for n in range(setval2.shape[0]):

                            dist = np.sum((setval1[m,:] - setval2[n,:])**2 / self.length_scale)
                            Kij_val = np.exp(-0.5*dist)

                            Kij += Kij_val

                    # normalize by product of cardinality of sets
                    Kij /= setval1.shape[0]*setval2.shape[0]
                    K[i,j] = Kij

            return K

    def diag(self, X):

        k_diag = []
        X = X.reshape(X.shape[0], self.num_set_size, self.num_feat_size)

        for i in range(X.shape[0]):
            
            # compute kset between each element of setval1 and setval2
            setval1 = X[i,:,:]
            setval2 = X[i,:,:]
            
            # ij'th element of the K matrix
            Kij = 0 

            # compute k_set
            for m in range(setval1.shape[0]):
                for n in range(setval2.shape[0]):
                    dist = np.sum((setval1[m,:] - setval2[n,:])**2 / self.length_scale)
                    Kij_val = np.exp(-0.5*dist)

                    Kij += Kij_val
                    
            
            # normalize by product of cardinality of sets
            Kij /= setval1.shape[0]*setval2.shape[0]

            k_diag.append(Kij)

        return np.asarray(k_diag)

    def __repr__(self):
        return "Set kernel implementation with basis RBF"