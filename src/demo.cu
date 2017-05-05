#include <time.h>
#include <string.h>
#include "idw.h"

int main(int argc, char **argv)
{
    float   *zValues, *zValuesGPU, *devZV, searchRadius;
    
    Point 	*knownPoints, *devKP;	
    Point2D *queryPoints, *devQP;	
    int KN, QN, sizeKP, sizeQP, stride, shMemSize, maxShmemSize, nIter, type; 	
    char *kp_filename, *loc_filename;
    
    // grid managing
    dim3 nBlocks, nThreadsForBlock;
    
    // gpu/cpu timing
    cudaEvent_t start, stop;
    float   cpuElapsedTime, gpuElapsedTime;
    clock_t cpuStartTime;

    cudaDeviceProp prop;

    cudaSetDevice(1);
    
    if (argc > 5)
    {
        type = atoi(argv[1]);
        nThreadsForBlock.x = atoi(argv[4]);
        searchRadius = atoi(argv[5]);
    }
    else
    {
        printf("\nUsage:\n");
        printf("./[bin_name] 1 [dataset_point_file] [query_locations_file] [block_threads_number] [search_radius]\n");
        printf("./[bin_name] 2 [known_values_number] [locations_number] [block_threads_number] [search_radius]\n\n");
	    exit(-1);
    }

    if (type == 1)
    {
        kp_filename = argv[2];
        loc_filename = argv[3];
        KN = getLines(kp_filename);
        QN = getLines(loc_filename);

        knownPoints = (Point*)malloc(KN*sizeof(Point));
        queryPoints = (Point2D*)malloc(QN*sizeof(Point2D));

        generateDataset(kp_filename, knownPoints);
        generateGrid(loc_filename, queryPoints);
    }
    else if (type == 2)
    {
        KN = atoi(argv[2]);
        QN = atoi(argv[3]);

        knownPoints = (Point*)malloc(KN*sizeof(Point));
        queryPoints = (Point2D*)malloc(QN*sizeof(Point2D));

        // generating random data for testing
        generateRandomData(knownPoints, queryPoints, 0, 1, KN, QN);
    }
    else 
    {
        printf("Type must be 1 or 2!\n");
        exit(-1);
    }

    sizeKP = KN*sizeof(Point);
    sizeQP = QN*sizeof(Point2D);
    nBlocks.x = ceil((float)QN/(float)nThreadsForBlock.x);

    cudaGetDeviceProperties(&prop,1);
    maxShmemSize = prop.sharedMemPerBlock/sizeof(Point);
    
    // known points are more than shared memory size?
    if (KN < maxShmemSize)
    {
        shMemSize = KN*sizeof(Point);
        nIter = 1;
        stride = ceil((float)KN/(float)nThreadsForBlock.x);
    }
    else
    {
        shMemSize = maxShmemSize*sizeof(Point);
        nIter = ceil((float)KN/(float)maxShmemSize);
        stride = ceil((float)maxShmemSize/(float)nThreadsForBlock.x);
    }
    
    zValues = (float*)malloc(QN*sizeof(float));
    zValuesGPU = (float*)malloc(QN*sizeof(float));

    cudaMalloc((void**)&devKP, sizeKP);
    cudaMalloc((void**)&devQP, sizeQP);
    cudaMalloc((void**)&devZV, QN*sizeof(float));

    printf("Data generated!\n\n");

    printf("Number of known points: %d\n", KN);
    printf("Number of query points: %d\n", QN);
    printf("Number of threads for block: %d\n", nThreadsForBlock.x);
    printf("Number of blocks: %d\n", nBlocks.x);
    printf("Stride: %d\n", stride);
    printf("Number of iterations: %d of max %d points\n\n", nIter, maxShmemSize);

    /* -- CPU -- */
    printf("Executing on CPU...\n");

    cpuStartTime = clock();
    
    if (sequentialIDW(knownPoints, queryPoints, zValues, KN, QN, searchRadius) < 0)
    {
        printf("Search radius is too small! Some values cannot be interpolated!\nYou need more dataset points or a different search radius!\n");
        exit(-1);
    }
    
    cpuElapsedTime = ((float)(clock() - cpuStartTime))/CLOCKS_PER_SEC;
    
    printf("Elapsed CPU time : %f s\n" ,cpuElapsedTime);
    /* --- END --- */

    /* -- GPU -- */
    printf("\nExecuting on GPU...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    
    cudaMemcpy(devKP, knownPoints, sizeKP, cudaMemcpyHostToDevice);
    cudaMemcpy(devQP, queryPoints, sizeQP, cudaMemcpyHostToDevice);

    parallelIDW<<<nBlocks,nThreadsForBlock,shMemSize>>>(devKP, devQP, devZV, KN, QN, stride, nIter, maxShmemSize, searchRadius);
    
    cudaMemcpy(zValuesGPU, devZV, QN*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuElapsedTime,start,stop);

    gpuElapsedTime = gpuElapsedTime*0.001;
    printf("Elapsed GPU time : %f s\n", gpuElapsedTime);

    checkCUDAError("parallelIDW");
    /* --- END --- */

    if (updateLogCpuGpu(gpuElapsedTime, cpuElapsedTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("\nLog updated\n");

    /*
    printf("Speed Up: %f\n\n", cpuElapsedTime/gpuElapsedTime);

    if (updateLog(gpuElapsedTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("Log updated\n");
    */

    printf("Residue: %e\n", getRes(zValues,zValuesGPU,QN));

    if (saveData(knownPoints, KN, queryPoints, zValues, zValuesGPU, QN, cpuElapsedTime, gpuElapsedTime) != -1)
        printf("\nResults saved!\n");
  
    free(knownPoints); free(queryPoints); free(zValues); free(zValuesGPU);
    cudaFree(devKP); cudaFree(devQP); cudaFree(devZV);
    
    return 0;
}
