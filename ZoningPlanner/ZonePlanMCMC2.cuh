#ifndef __ZONE_PLAN_MCMC2_CUH__


#define CITY_SIZE 10 //200
#define CITY_CELL_LEN 400 //20
#define GPU_BLOCK_SIZE 20//40
#define GPU_NUM_THREADS 1//96
#define GPU_BLOCK_SCALE (1.0)//(1.1)
#define NUM_FEATURES 5
#define QUEUE_MAX 1999
#define MAX_DIST 99

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 


struct zone_type {
	int type;
	int level;
};

struct zone_plan {
	zone_type zones[CITY_SIZE][CITY_SIZE];
};

struct DistanceMap {
	int distances[CITY_SIZE][CITY_SIZE][NUM_FEATURES];
};


void zonePlanMCMCGPUfunc2(zone_plan** bestPlan, int numIterations);

#endif
