# Top_FCNC_Lightning 
## Authors: $$\textcolor{blue}{\text{Benjamin Fuks,  Sumit K. Garg, A. Hammad and Adil Jueid}}$$ 
&emsp; DL codes used for the Top quark FCNC analysis based on [arXiv:xxx](https://arxiv.org/abs/2207.09959). 
 __________
 ## $~~~~~~~~~~~$  Table of content

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [1. Introduction ](#Introduction)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [2. Prerequisites ](#Prerequisites)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [3. Package structure ](#structure)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [4. Get started ](#start)




 <a name="Introduction"></a>
## Introduction
&emsp;  This code implements attention based Transformer and GCN network to analyze the signatures of the Top quark FCNC at the LHC. Inputs to the Transformer is a particle cloud structure while for the GCN a fully connected graph. ...............


<a name="Requirements"></a>
##  Prerequisites
&emsp; To run the package you need python3 with the following modules:
* **Clone the Repository:**  Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/AHamamd150/Top_FCNC_Lightning/.git
cd Top_FCNC_Lightning
```
* **Install Conda:** If you do not have Miniconda or Anaconda installed, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) respectively.
  
* **Set Up Your Environment:**   This project relies on several dependencies listed in `environment.yml`, including libraries such as NumPy, Pandas, Matplotlib, tqdm, h5py, scikit-learn, PyTorch, PyTorch Geometric, PyTorch Lightning, and Torchmetrics. To install all dependencies at once and create a Conda environment named `higgscp`, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

* **Activate the Environment:**
  ```bash
conda activate higgscp
```


