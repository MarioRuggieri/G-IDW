#include <time.h>
#include <string.h>
#include "idw.h"

#define N_ITER 25

int main(int argc, char **argv)
{
    float   *zValues, *zValuesGPU, *devZV, maxErr;
    
    Point 	*knownPoints, *devKP;	
    Point2D *queryPoints, *devQP;	
    int KN, QN, sizeKP, sizeQP, stride, shMemSize, nIter; 	
    
    // grid managing
    dim3 nBlocks, nThreadsForBlock;
    
    // gpu/cpu timing
    cudaEvent_t start, stop;
    float   cpuElapsedTime[N_ITER], cpuMeanTime, cpuSTD, 
            gpuElapsedTime[N_ITER], gpuMeanTime, gpuSTD;
    clock_t cpuStartTime;
    
    if (argc > 3)
    {
        KN = atoi(argv[1]);
        QN = atoi(argv[2]);
        nThreadsForBlock.x = atoi(argv[3]);
        nBlocks.x = ceil((float)QN/(float)nThreadsForBlock.x);
    }
    else
    {
        printf("\nUsage:\n\n ./[bin_name] [known_points_number] [locations_number] [block_threads_number]\n\n");
	exit(-1);
    }
    
    sizeKP = KN*sizeof(Point);
    sizeQP = QN*sizeof(Point2D);
    
    // known points are more than shared memory size?
    if (KN < MAX_SHMEM_SIZE)
    {
        shMemSize = KN*sizeof(Point);
        nIter = 1;
        stride = ceil((float)KN/(float)nThreadsForBlock.x);
    }
    else
    {
        shMemSize = MAX_SHMEM_SIZE*sizeof(Point);
        nIter = ceil((float)KN/(float)MAX_SHMEM_SIZE);
        stride = ceil((float)MAX_SHMEM_SIZE/(float)nThreadsForBlock.x);
    }
    
    knownPoints = (Point*)malloc(sizeKP);
    queryPoints = (Point2D*)malloc(sizeQP);
    zValues = (float*)malloc(QN*sizeof(float));
    zValuesGPU = (float*)malloc(QN*sizeof(float));

    cudaMalloc((void**)&devKP, sizeKP);
    cudaMalloc((void**)&devQP, sizeQP);
    cudaMalloc((void**)&devZV, QN*sizeof(float));

    // generating random data for testing
    generateRandomData(knownPoints, queryPoints, 0, 100, KN, QN);
    
    printf("Data generated!\n\n");

    printf("Number of known points: %d\n", KN);
    printf("Number of query points: %d\n", QN);
    printf("Number of threads for block: %d\n", nThreadsForBlock.x);
    printf("Number of blocks: %d\n", nBlocks.x);
    printf("Stride: %d\n", stride);
    printf("Number of iterations: %d of max %d points\n\n", nIter, MAX_SHMEM_SIZE);

    /* -- CPU -- */
    cpuMeanTime = 0;
    for (int j=0; j<N_ITER; j++)
    {
        cpuStartTime = clock();
    
        sequentialIDW(knownPoints, queryPoints, zValues, KN, QN);
    
        cpuElapsedTime[j] = ((float)(clock() - cpuStartTime))/CLOCKS_PER_SEC;
    
        printf("Elapsed CPU time : %f s\n" ,cpuElapsedTime[j]);

        cpuMeanTime += cpuElapsedTime[j];
    }

    cpuMeanTime /= N_ITER;
    cpuSTD = getSTD(cpuMeanTime, cpuElapsedTime, N_ITER);

    printf("Elapsed CPU MEAN time : %f s\n", cpuMeanTime);
    printf("CPU std: %f\n\n", cpuSTD);
    /* --- END --- */

    /* -- GPU -- */
    gpuMeanTime = 0;
    for (int j=0; j<N_ITER; j++)
    {

        /*cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaEventCreate(&event);*/

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
    
        cudaMemcpy(devKP, knownPoints, sizeKP, cudaMemcpyHostToDevice);//, stream1);
        //cudaEventRecord(event, stream1);

        cudaMemcpy(devQP, queryPoints, sizeQP, cudaMemcpyHostToDevice);//, stream2);
        //cudaStreamWaitEvent(stream2, event, 0);

        parallelIDW<<<nBlocks,nThreadsForBlock,shMemSize>>>(devKP, devQP, devZV, KN, QN, stride, nIter);
    
        cudaMemcpy(zValuesGPU, devZV, QN*sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuElapsedTime[j],start,stop);

        gpuElapsedTime[j] = gpuElapsedTime[j]*0.001;
        printf("Elapsed GPU time : %f s\n", gpuElapsedTime[j]);

        checkCUDAError("parallelIDW");

        gpuMeanTime += gpuElapsedTime[j];
    }
    /* --- END --- */

    gpuMeanTime /= N_ITER;
    gpuSTD = getSTD(gpuMeanTime, gpuElapsedTime, N_ITER);

    printf("Elapsed GPU MEAN time : %f s\n", gpuMeanTime);
    printf("GPU std: %f\n", gpuSTD);

    if (updateLogCpuGpu(gpuMeanTime, cpuMeanTime, gpuSTD, cpuSTD, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("\nLog updated\n");

    /*
    printf("Speed Up: %f\n\n", cpuElapsedTime/gpuElapsedTime);

    if (updateLog(gpuMeanTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("Log updated\n");
    */

    getMaxAbsError(zValues, zValuesGPU, QN, &maxErr);
    printf("Max abs error: %f\n", maxErr);

    /*
    if (saveData(knownPoints, KN, queryPoints, zValues, zValuesGPU, QN, cpuElapsedTime, gpuElapsedTime) != -1)
        printf("\nResults saved!\n");
    */
  
    free(knownPoints); free(queryPoints); free(zValues); free(zValuesGPU);
    cudaFree(devKP); cudaFree(devQP); cudaFree(devZV);
    
    return 0;
}
