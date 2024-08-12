#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void findNearest(int *xcoord, int *ycoord, int *hp, int *score, int round, int T, int M, int N, int *remainingTanks, int *nearest_arr) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    __shared__ int dist2;
    __shared__ int dirTarget;
    __shared__ int shooterX;
    __shared__ int shooterY;
    __shared__ int targetX;
    __shared__ int targetY;
    __shared__ int quadrant1;

    if(j == 0) {
        dist2 = INT_MAX;
    }
    __syncthreads();
    
    if (j == 0 && hp[i] > 0) {
        dirTarget = (i + round) % T;
        shooterX = xcoord[i];
        shooterY = ycoord[i];
        targetX = xcoord[dirTarget];
        targetY = ycoord[dirTarget];
        if (targetX >= shooterX && targetY >= shooterY)
            quadrant1 = 1;
        else if (targetX <= shooterX && targetY >= shooterY)
            quadrant1 = 2;
        else if (targetX <= shooterX && targetY <= shooterY)
            quadrant1 = 3;
        else
            quadrant1 = 4;
    }
    __syncthreads();

    if(hp[i] > 0) {
        int quadrant2;
        if (xcoord[j] >= shooterX && ycoord[j] >= shooterY)
            quadrant2 = 1;
        else if (xcoord[j] <= shooterX && ycoord[j] >= shooterY)
            quadrant2 = 2;
        else if (xcoord[j] <= shooterX && ycoord[j] <= shooterY)
            quadrant2 = 3;
        else
            quadrant2 = 4;

        if((j != i && hp[j] > 0 && quadrant1 == quadrant2)) {
            int dx = xcoord[j] - shooterX;
            int dy = ycoord[j] - shooterY;
            
            if(((targetY - shooterY) * dx) == (dy * (targetX - shooterX))) {
                int dist = abs(dx) + abs(dy);
                atomicMin(&dist2, dist);
            }
        }
    }
    __syncthreads();

    if(hp[i] > 0) {
        int quadrant2;
        if (xcoord[j] >= shooterX && ycoord[j] >= shooterY)
            quadrant2 = 1;
        else if (xcoord[j] <= shooterX && ycoord[j] >= shooterY)
            quadrant2 = 2;
        else if (xcoord[j] <= shooterX && ycoord[j] <= shooterY)
            quadrant2 = 3;
        else
            quadrant2 = 4;

        if((j != i && hp[j] > 0 && quadrant1 == quadrant2)) {
            int dx = xcoord[j] - shooterX;
            int dy = ycoord[j] - shooterY;
            
            if(((targetY - shooterY) * dx) == (dy * (targetX - shooterX))) {
                int dist = abs(dx) + abs(dy);
                if(dist2 == dist) {
                    nearest_arr[i] = j;
                }
            }
        }
    }
}

__global__ void calculateScore(int *nearest, int *hp, int *remainingTanks, int *score) {
    int i = threadIdx.x;
    if(hp[i] <= 0) return;
    __syncthreads();
    if (nearest[i] != -1) {
        if(atomicAdd(&hp[nearest[i]], -1) == 1) {
            atomicAdd(remainingTanks, -1);
        }
        score[i]++;
    }
}

//***********************************************

int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++) {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *h_hp;
    h_hp = (int *)malloc(T * sizeof(int));
    for (int i = 0; i < T; i++) {
        h_hp[i] = H;
    }

    int *g_xcoord;
    int *g_ycoord;
    int *g_hp;

    cudaMalloc(&g_xcoord, T * sizeof(int));
    cudaMalloc(&g_ycoord, T * sizeof(int));
    cudaMalloc(&g_hp, T * sizeof(int));

    cudaMemcpy(g_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_hp, h_hp, T * sizeof(int), cudaMemcpyHostToDevice);

    int *g_score;
    cudaMalloc(&g_score, T * sizeof(int));

    // Initialize score array to 0
    cudaMemset(g_score, 0, T * sizeof(int));

    int round = 1;
    int *remainingTanks = (int *)malloc(sizeof(int));
    int *g_remainingTanks;
    int *g_nearest_arr;

    *remainingTanks = T;

    cudaMalloc(&g_remainingTanks, sizeof(int));
    cudaMalloc(&g_nearest_arr, T * sizeof(int));
    cudaMemcpy(g_remainingTanks, remainingTanks, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(g_nearest_arr, -1, T * sizeof(int));

    while (*remainingTanks > 1) {
        if (round % T != 0) {
            findNearest<<<T, T>>>(g_xcoord, g_ycoord, g_hp, g_score, round, T, M, N, g_remainingTanks, g_nearest_arr);
            calculateScore<<<1, T>>>(g_nearest_arr, g_hp, g_remainingTanks, g_score);
        }
        cudaMemcpy(remainingTanks, g_remainingTanks, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(g_nearest_arr, -1, T * sizeof(int));
        round++;
    }

    cudaMemcpy(score, g_score, T * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(g_xcoord);
    cudaFree(g_ycoord);
    cudaFree(g_hp);
    cudaFree(g_score);
    cudaFree(g_remainingTanks);
    cudaFree(g_nearest_arr);
  
    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    
    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++) {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);
    
    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}