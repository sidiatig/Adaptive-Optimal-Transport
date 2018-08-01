# Adaptive-Optimal-Transport
Code for the papers based on arxiv.org/pdf/1807.00393.pdf.

## Main files
* AOT.ipynb : Jupyter notebook to visualize data, code and debug the main functions. Main file.
* GH.py : Computation of twisted Gradient and Hessian.
* local_ot.py : Solves the local sample based optimal transport using an implicit gradient descent method.
* global_ot.py : Solves the sample based optimal transport problem by solving multiple local problems for the forward step, and use displacement interpolation for the backward step.

## Some pictures
### Gaussian to ring
![Gaussian to ring](/pictures/ex2start.png "Initial configuration")![](/pictures/ex2end.png "Final configuration")

### Gaussian to mixture of 2 Gaussians
![Gaussian to mixture of 2 Gaussians](/pictures/ex3start.png "Initial configuration")![](/pictures/ex3end.png "Final configuration")

## Coding Standards
* Clearly indicate the inputs and outputs (+type, dimensions) of a function, as well as a brief description, in a comment at the beginning (use '''comment''')
* When applicable, clearly delimitate different sub-procedures inside a function by a comment of a brief description (use #comment at the beginning of a paragraph)
* Use explicit names of variables

## To Do 

### Table (last update 2018/07/31)

| Task | Person |     State    |
|:----:|:------:|:------------:|
|   a  |    x   |     Done     |
|   b  |    y   |   Ongoing    |
|   c  |    z   |  Not Started |

### Task summary
* a:
* b:
* c:
