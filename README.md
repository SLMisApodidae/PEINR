# PEINR
A INR method for High-Fidelity Flow Field Reconstruction

A PyTorch implementation of PEINR based on ICML2025 paper: PEINR: A Physics-enhanced Implicit Neural Representation for High-Fidelity
Flow Field Reconstruction.

# requirement
- python==3.x (Let's move on to python 3 if you still use python 2)
- pytorch==2.0.0
- numpy>=1.15.4
- sentencepiece==0.1.8
- tqdm>=4.28.1

# dataset
The volume at each time step is saved as a .dat or .plt file with the little-endian format. The data is stored in row-major order, that is, x-axis goes first, then y-axis, finally z-axis. The low-resolution and high-resolution volumes are both simulation data.

https://www.alipan.com/t/DoU8ifv04AFjqsIXSk7w


<img width="527" alt="image" src="https://github.com/user-attachments/assets/d528c224-9e08-4495-bfda-7d7c958f3583" />
<img width="522" alt="image" src="https://github.com/user-attachments/assets/7cb10890-5333-43f5-822e-48c8d840c219" />
<img width="526" alt="image" src="https://github.com/user-attachments/assets/26b372c0-82d4-4f71-8cdc-d462281e76e6" />


# train
first change the data path in dataio.py, then 
`python main.py --train`

`
# inference
`python main.py --inf`
