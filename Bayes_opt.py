# %%

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from SetKernel import SetKernel
import itertools
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os

# %%

###############################
# Black-box objective function
###############################
def objective(x):
    x1 = x[0]
    x2 = x[1]
    
    # Branin function 
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    
    func_val = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s

    return -1*func_val
 
def surrogate(model, X):
    
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")

        return model.predict(X, return_std=True)

#######################
# Acquisition function
#######################
def acquisition(X, Xsamples, model):

    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    std = np.expand_dims(std, 1)

    # Expected Improvement
    with np.errstate(divide='warn'):
        imp = mu - best - 0.01
        Z = imp / std 
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei

################################
# OPTIMIZE ACQUISITION FUNCTION
################################
def opt_acquisition(X, y, model, X_total, y_total):

    # random search, generate random samples
    randidx = np.random.choice(X_total.shape[0], size=3000)
    Xsamples = X_total[randidx,:]

    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)

    # locate the index of the largest scores
    ix = np.argmax(scores)
    
    ret_val = Xsamples[ix,:].reshape(1,2)

    return ret_val

####################
# PLOTTING FUNCTION
####################
def plot_contour(X_train, model, iter_idx=0, plotprior=False): 

    # BRANIN FUNCTION RANGE

    x1 = list(np.arange(-5, 10, 0.1))
    x2 = list(np.arange(0, 15, 0.1))

    xx, yy = np.meshgrid(x1, x2)
   
    pred_input = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[0]):
            pred_input[i,j] = model.predict(np.array([[xx[i,j], yy[i,j]]]))

    min_val = np.min(-1*pred_input)
    max_val = np.max(-1*pred_input)

    # PLOT SURFACE PLOT 

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, -1*pred_input, vmin=min_val, vmax=max_val, cmap='inferno')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f({ [x1, x2] })')

    if iter_idx==0:
        file_name = "./plots/prior_3D_mesh.png"
    else:
        file_name = "./plots/iter_{}_3D_mesh.png".format(iter_idx)

    fig.savefig(file_name, dpi=700)
   
   # CONTOUR PLOT 

    plt.figure()
    plt.contourf(xx, yy, -1*pred_input, vmin=min_val, vmax=max_val, cmap='inferno')
    plt.xlabel("x1")
    plt.ylabel("x2")

    # PLOT GLOBAL MINIMA OF FUNCTION 

    plt.scatter(-np.pi, 12.275, marker='^', color="#FFA500", label="global minima")
    plt.scatter(np.pi, 2.275, marker='^', color="#FFA500")
    plt.scatter(9.42478, 2.475, marker='^', color="#FFA500")

    # PLOT B.O. SUGGESTIONS 

    if plotprior:
        plt.scatter(X_train[:, 0], X_train[:, 1], color="#0000FF")
    else:
        plt.scatter(X_train[-1*np.arange(1,6), 0], X_train[-1*np.arange(1,6), 1], color="#FF0000", label="last 5 points suggested by B.O.")
    
    # last point predicted by B.O.
    plt.scatter(X_train[-1,0], X_train[-1,1] , color="#00FF00", marker='x', label="previous point predicted by B.O.")
    
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_clim(min_val, max_val)
    plt.colorbar(m, boundaries=np.linspace(min_val, max_val, 1000))

    if iter_idx==0:
        file_name = "./plots/prior_2D_contour.png"
    else:
        file_name = "./plots/iter_{}_2D_contour.png".format(iter_idx)

    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    plt.legend()
    plt.show()

# %%
################
# GENERATE DATA
################

# define function range
x1 = list(np.arange(-5, 10, 0.1)) # branin range - x1: (-5, 10) x2: (0, 15)
x2 = list(np.arange(0, 15, 0.1))

# generate data
coord_tuples = list(itertools.product(x1, x2))
X_total = []
y_total = []
for tup in coord_tuples:
    tuplist = list(tup)
    y_total.append(objective(tuplist))
    X_total.append(tuplist)

X_total = np.asarray(X_total)
y_total = np.asarray(y_total)

'''
If size of domain is (N, num_elem_in_set, num_feat),  
bring it to  (N, num_elem_in_set*num_feat)
This is resized back into (N, num_elem_in_set, num_feat) 
inside the Set kernel for Cov computation.
'''

# select N points to create prior. 
randidx = np.random.choice(X_total.shape[0], size=10)
X_gp = X_total[randidx,:]
y_gp = y_total[randidx]

y_total = y_total.reshape(y_total.shape[0], 1)
y_gp = y_gp.reshape(y_gp.shape[0], 1)

# %%

######################
# FIT GP TO GET PRIOR
######################

# initialize our custom kernel
newKernel = SetKernel(num_set_size=1, num_feat_size=2)

# fit GP
model = GaussianProcessRegressor(kernel=newKernel, normalize_y=True)
model.fit(X_gp, y_gp)

# get 3D and 2D contour plots and save them 
if not os.path.exists('./plots'):
    os.makedirs('./plots')

plot_contour(X_gp, model, plotprior=True)

# %%

########################
# PERFORM B.O. 
#######################

for i in range(1, 100):

    # select the next point to sample    
    x = opt_acquisition(X_gp, y_gp, model, X_total, y_total)

    # get ground-truth
    actual = objective(x[0])
    
    # summarize the finding
    est, _ = surrogate(model, x)
    print('>x=',x,', f()=%3f, actual=%.3f' % (est, actual))
    
    # add the data to the dataset
    X_gp = np.vstack((X_gp, x))
    y_gp = np.vstack((y_gp, [[actual]]))
    
    # update the model
    model.fit(X_gp, y_gp)
    print("GP length scale is ", model.kernel_.length_scale)
    if i % 5==0:
        plot_contour(X_gp, model, iter_idx=i)


# output function optima
ix = np.argmax(y_gp)
print('Best Result: x=',X_gp[ix],' y=',y_gp[ix])

# %%
