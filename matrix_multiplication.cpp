#include <iostream>
#include <chrono>
#include <ctime>
#include <random>
#include <cblas.h>
#include <omp.h>

int main()
{
    // declare and initialize A, B and N
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist(0, 1);
    const int N = 1024;
    double* A = (double*) malloc(sizeof(double)*N*N);
    double* B = (double*) malloc(sizeof(double)*N*N);

    for(uint i = 0; i < N; i++)
        for(uint j = 0; j < N; j++){
            A[i*N+j] = doubleDist(rnd);
            B[i*N+j] = doubleDist(rnd);
        }

    // declare C
    double* C = (double*) malloc(sizeof(double)*N*N);

    // simple multiplication
    std::cout << "############## simple multiplication #############\n";
    // set start time
    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            C[i*N+j] = 0;
            for(int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
        }

    // display result of multplication
    std::cout << "A * B = C est terminé\n";

    // set end time
    auto end = std::chrono::system_clock::now();

    // convert end time to display it
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";


    // OpenMP optimized multiplication
    std::cout << "############## optimized multiplication using OpenMP #############\n";

    // set start time
    start = std::chrono::system_clock::now();


#pragma omp parallel for
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
        {
            C[i*N+j] = 0;
            for(int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
        }
    }

    // Cblas multiplication
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);


    // display result of multplication
    std::cout << "A * B = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";


    // Transposition optimized multiplication
    std::cout << "############## optimized multiplication using Transposition #############\n";

    // set start time
    start = std::chrono::system_clock::now();

    // transpose B to B2
    double* B2 = (double*) malloc(sizeof(double)*N*N);
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            B2[ j*N + i ] = B[ i*N + j ];


    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
        {
            C[i*N+j] = 0;
            for(int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B2[j*N+k];
        }
    }

    // Cblas multiplication
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);


    // display result of multplication
    std::cout << "A * B2 = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";
    // OpenMP + Transposition optimized multiplication
    std::cout << "############## optimized multiplication using OpenMP + Transposition #############\n";

    // set start time
    start = std::chrono::system_clock::now();


#pragma omp parallel for
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
        {
            C[i*N+j] = 0;
            for(int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B2[j*N+k];
        }
    }

    // Cblas multiplication
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);


    // display result of multplication
    std::cout << "A * B2 = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";

    // Tiling optimized multiplication
    std::cout << "############## optimized multiplication using Tiling #############\n";

    // set start time
    start = std::chrono::system_clock::now();

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            C[i*N + j] = 0;


    for (int i0=0; i0 < N/32; i0++)
        for (int j0=0; j0 < N/32; j0++)
            for (int k0=0; k0 < N/32; k0++)
                for (int i1=32*i0; i1<32*i0+32; i1++)
                    for (int j1=32*j0; j1<32*j0+32; j1++)
                        for (int k1=32*k0; k1<32*k0+32; k1++)
                            C[i1*N + j1] += A[i1*N + k1] * B[k1*N + j1];


    // display result of multplication
    std::cout << "A * B (tiling) = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";

    // Transposition + Tiling optimized multiplication
    std::cout << "############## optimized multiplication using Transposition + Tiling #############\n";

    // set start time
    start = std::chrono::system_clock::now();

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            C[i*N + j] = 0;


    for (int i0=0; i0 < N/32; i0++)
        for (int j0=0; j0 < N/32; j0++)
            for (int k0=0; k0 < N/32; k0++)
                for (int i1=32*i0; i1<32*i0+32; i1++)
                    for (int j1=32*j0; j1<32*j0+32; j1++)
                        for (int k1=32*k0; k1<32*k0+32; k1++)
                            C[i1*N + j1] += A[i1*N + k1] * B2[j1*N + k1];



    // display result of multplication
    std::cout << "A * B2 (with tiling) = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";

    // OpenMP + Transposition + Tiling optimized multiplication
    std::cout << "############## optimized multiplication using OpenMP + Transposition + Tiling #############\n";

    // set start time
    start = std::chrono::system_clock::now();

#pragma omp parallel for
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            C[i*N + j] = 0;


#pragma omp parallel for
    for (int i0=0; i0 < N/32; i0++)
        for (int j0=0; j0 < N/32; j0++)
            for (int k0=0; k0 < N/32; k0++)
                for (int i1=32*i0; i1<32*i0+32; i1++)
                    for (int j1=32*j0; j1<32*j0+32; j1++)
                        for (int k1=32*k0; k1<32*k0+32; k1++)
                            C[i1*N + j1] += A[i1*N + k1] * B2[j1*N + k1];


    // display result of multplication
    std::cout << "A * B2 (with OpenMP + Tiling) = C est terminé\n";
    // set end time
    end = std::chrono::system_clock::now();


    // convert end time to display it
    end_time = std::chrono::system_clock::to_time_t(end);
    // calcul the elapsed time in sec
    elapsed_seconds = end-start;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n\n";
}
