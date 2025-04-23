# Top_FCNC_Lightning 
## Authors: $$\textcolor{blue}{\text{Benjamin Fuks,  Sumit K. Garg, A. Hammad and Adil Jueid}}$$ 
&emsp; DL codes used for the Top quark FCNC analysis based on [arXiv:xxx](https://arxiv.org/abs/2207.09959). 
 __________
 ## $~~~~~~~~~~~$  Table of content

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [1. Introduction ](#Introduction)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [2. Prerequisites ](#Prerequisites)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [3. Code structure ](#structure)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [4. Get started ](#start)




 <a name="Introduction"></a>
## Introduction
&emsp;  This code implements attention based Transformer and GCN network to analyze the signatures of the Top quark FCNC at the LHC. Inputs to the Transformer is a particle cloud structure while for the GCN a fully connected graph. ...............

<span>
  <img width="635" style="display:inline-block; vertical-align:top;" src="https://github.com/user-attachments/assets/0e5b442a-2ea2-4a88-83de-638b0a4bc5b5" />
</span>
<span>
  <img width="505" style="display:inline-block; vertical-align:top;" src="https://github.com/user-attachments/assets/756135fb-0147-44be-8c9a-76d6220a4675" />
</span>


<a name="Requirements"></a>
##  Prerequisites
&emsp; To run the package you need python3 with the following modules:
* **Clone the Repository:**  Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/AHamamd150/Top_FCNC_Lightning/.git
cd Top_FCNC_Lightning
```
* **Install Conda:** If you do not have Miniconda or Anaconda installed, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) respectively.
  
* **Set Up Your Environment:**   This project relies on several dependencies listed in `environment.yml`, including libraries such as NumPy, Pandas, Matplotlib, tqdm, h5py, scikit-learn, PyTorch, PyTorch Geometric, PyTorch Lightning, and Torchmetrics. To install all dependencies at once and create a Conda environment named `Top_FCNC`, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

* **Activate the Environment:**

```bash
conda activate Top_FCNC
```

<a name="structure"></a>
##  Code structure


<a name="start"></a>
##  Get started
To traint the Transformer network.
```bash
python main.py fit --config config/config_Transformer.yaml
```
To traint the Graph Attention network.
```bash
python main.py fit --config config/config_GAT.yaml
```

For testing the network one need to retore the weigths and the configuration file from the best epoch results as 

```bash
python main.py test -c checkpoints/version_0/config.yaml --ckpt_path checkpoints/version_0/checkpoints/best_checkpoint.ckpt
```
