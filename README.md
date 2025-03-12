# Top_FCNC_Lightning 
## Authors: $$\textcolor{blue}{\text{Benjamin Fuks,  Sumit K. Garg, A. Hammad and Adil Jueid}}$$ 
&emsp; DL codes used for the Top quark FCNC analysis based on [arXiv:xxx](https://arxiv.org/abs/2207.09959). 
 __________
 ## $~~~~~~~~~~~$  Table of content

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [1. Introduction ](#Introduction)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [2. Requirements ](#Requirements)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [3. Package structure ](#structure)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [4. Get started ](#start)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 <a name="Introduction"></a>
## Introduction
&emsp;  This code implements attention based Transformer and GCN network to analyze the signatures of the Top quark FCNC at the LHC. Inputs to the Transformer is a particle cloud structure while for the GCN a fully connected graph. ...............


<a name="Requirements"></a>
## Requirements
&emsp; To run the package you need python3 with the following modules:
* Numpy
* TensorFlow
* sklearn
* imblearn 
* multiprocessing (for the intial training over multi-cores)
* tqdm (for the illustration of the fancy progress bar)

Requirements can be easily installed by `pip3 install module`
