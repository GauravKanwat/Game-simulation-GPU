# Game Simulation with GPU Acceleration

This project contains a game simulation logic that utilizes GPU acceleration for parallel operations. The core logic is implemented in the `CS23M024.cu` file using CUDA.

## Description

The game simulation leverages GPU to handle computations parallely. The simulation logic is written in CUDA C and is designed to run on NVIDIA GPUs. The file utilizes parallel processing to perform tank operations, such as shooting different tanks and vanishing tanks, simultaneously. Additionally, the tanks are represented as coordinates in a graph.

## Installation

To set up the environment for running this simulation, follow these steps:

1. **Install CUDA Toolkit**: Ensure you have the CUDA Toolkit installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

2. **Clone the Repository**: Clone this repository to your local machine.
    ```sh
    git clone https://github.com/GauravKanwat/Game-simulation-GPU
    cd Game-simulation-GPU
    ```

3. **Compile the CUDA Code**: Use the `nvcc` compiler to compile the CUDA code.
    ```sh
    nvcc -o game_simulation CS23M024.cu
    ```

## Usage

To run the game simulation, execute the compiled binary:
```sh
./game_simulation