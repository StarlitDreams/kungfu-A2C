
<img width="3942" height="1536" alt="Banner" src="https://github.com/user-attachments/assets/cdd85300-e1be-4daa-8943-b830c8c3cd01" />

## Overview
This repository implements an Advantage Actor–Critic (A2C) agent trained to play the Atari game **Kung Fu Master**. The agent was developed in PyTorch and uses Gymnasium (with ALE ROMs registered) for environment management. The main training and evaluation script is provided in **`main.py`**.

## Repository Structure
- **`main.py`**  
  Implements environment setup, preprocessing, the neural network, the A2C agent, batch training loop, evaluation, and video recording/display utilities.
- **`requirements.txt`**  
  Lists all Python dependencies.

## Prerequisites
- **Python** 3.8 or higher  
- **CUDA-capable GPU** (optional but recommended for faster training)  
- **Git** (to clone this repository)

## Installation
- **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/kungfu-a2c.git
   cd kungfu-a2c
   ```
- **Create and activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # on macOS/Linux
   .venv\Scripts\activate.bat   # on Windows
   ```
- **Install dependencies**
 ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```
## Usage

### Training
Run the main script to start training the A2C agent:
```bash
python main.py
```
- Training will be performed for 300 000 environment steps by default.  
- Every 1 000 steps, the agent will be evaluated over 10 episodes and the average reward will be printed.  
- At the end of training, **`video.mp4`** will be generated in the working directory.


## Training Iterations
300 000 iterations were done which took 6 hours to complete.
<img width="1920" height="1080" alt="Screenshot 2025-07-21 141318" src="https://github.com/user-attachments/assets/1c0eecff-3fc3-43eb-9f77-20e227d60088" />


## Game Demo

A recorded demonstration of the trained agent playing **Kung Fu Master**.

![video](https://github.com/user-attachments/assets/31a64efc-0e4a-46fb-8838-3f470b244503)



## Environment & Preprocessing

- **`gymnasium`** was used for Atari environment management.  
- A custom `PreprocessAtari` wrapper:
  - Resizes frames to 42×42  
  - Converts to grayscale  
  - Normalizes pixel values to [0, 1]  
  - Stacks 4 consecutive frames
 
 
## A2C Agent Details

- **Input**: stack of 4 grayscale frames (42×42).  
- **Network architecture**:  
  - **Conv1**: 4→32 channels, 3×3 kernel, stride 2  
  - **Conv2**: 32→64 channels, 3×3 kernel, stride 2  
  - **Conv3**: 64→64 channels, 3×3 kernel, stride 2  
  - **Flatten** → **FC** (128 units) → two heads:  
    - **Policy** logits (action_size)  
    - **Value** estimate (1), squeezed  
- **Reward scaling**: rewards are multiplied by 0.01 before computing targets.  
- **Discount factor (γ)**: 0.99  
- **Optimizer**: Adam (learning rate = 1 × 10⁻⁴)  
- **Losses**:  
  - **Policy loss**: \(-\mathbb{E}[\log \pi(a|s)\,\times\,\text{advantage}]\;-\;0.001\times\text{entropy}\)  
  - **Value loss**: mean squared error between predicted value and TD-target  
- **Entropy coefficient**: 0.001  



