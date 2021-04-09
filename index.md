# Bayesian-Optimization-on-Sets

## Dependencies and instructions
1. scikit-learn
2. matplotlib
3. numpy 
4. scipy 

```python 
python Bayes_opt.py
```
## Introduction and results
Bayesian optimization is used to approximate black-box functions that are expensive to evaluate. It has proven useful in several applications, including hyperparameter optimization neural architecture search and material design. Classic BO assumes a search region ![equation](https://latex.codecogs.com/gif.latex?X%20%5Csubset%20%5Cmathcal%7BR%7D%5E%7Bd%7D) and a scalar black-box function f evaluated in the presence of additive noise. 

Unlike this standard BO formulation, we assume that our search region is ![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BX%7D_%7Bset%7D%20%3D%20%5C%7B%5C%7Bx_%7B1%7D%2C...%2Cx_%7Bm%7D%5C%7D%20%7C%20x_%7Bi%7D%20%5Cin%20%5Cmathcal%7BX%7D%20%5Csubset%20%5Cmathbb%7BR%7D%5E%7Bd%7D%5C%7D) for a fixed positive integer m. Thus, for ![equation](https://latex.codecogs.com/gif.latex?X%20%5Cin%20%5Cmathcal%7BX%7D_%7Bset%7D), f would take in a "set" containing m elements, all of length d, and return a noisy function value ![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20f%28X%29%20&plus;%20%5Cepsilon)

We specifically implement equation 4 in the paper [Practical B.O. over Sets](https://arxiv.org/pdf/1905.09780.pdf) which is basically a Set Kernel. To illustrate the algorithm, we take a standard non-convex test function, [Branin function](https://www.sfu.ca/~ssurjano/branin.html), with multiple optima. Furthermore, we discretize the domain into coordinate tuples (x,y). We treat each of these tuples as a set a with cardinality 1 and dimesnion 2. Our entire domain is now a set of sets. Gaussian Process (GP) is used as the surrogate model and Expected improvement acquisition function is used. We make use of our custom SetKernel as the kernel for GP. 

In the following GIFs we illustrate the performance of the B.O over sets algorithm using contour plots and surface plots. The B.O is run for 100 iterations and the plots in the GIFs are taken every 5 iterations. The orange triangles represent the global optima of the function. The red dots represent the last 5 points suggested by the B.O. The green X represents the last point suggested by B.O. We can notice that as the iterations proceed, the points being suggested are more closer to the global optima of the function which shows expected behaviour. From the mesh plot we can visualize how the function is getting learned over the course of the iterations. 

![alt text](https://github.com/kaushik333/Bayesian-Optimization-on-Sets/blob/main/readme_stuff/contour_gif.gif "Contour plots")
![alt text](https://github.com/kaushik333/Bayesian-Optimization-on-Sets/blob/main/readme_stuff/mesh_gif.gif "Mesh surface plots")

