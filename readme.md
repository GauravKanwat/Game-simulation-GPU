# Game Simulation with GPU Acceleration

This project contains a game simulation logic that utilizes GPU acceleration for improved performance. The core logic is implemented in the `CS23M024.cu` file using CUDA.

## Description

The game simulation leverages the power of GPU to handle complex computations efficiently. This results in faster processing times and smoother gameplay experience. The simulation logic is written in CUDA C and is designed to run on NVIDIA GPUs.

## Installation

To set up the environment for running this simulation, follow these steps:

1. **Install CUDA Toolkit**: Ensure you have the CUDA Toolkit installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

2. **Clone the Repository**: Clone this repository to your local machine.
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

3. **Compile the CUDA Code**: Use the `nvcc` compiler to compile the CUDA code.
    ```sh
    nvcc -o game_simulation CS23M024.cu
    ```

## Usage

To run the game simulation, execute the compiled binary:
```sh
./game_simulation