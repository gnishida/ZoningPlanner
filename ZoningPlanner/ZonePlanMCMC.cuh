#ifndef __ZONE_PLAN_MCMC_CUH__

#include <stdio.h>

#define ZONE_PLAN_MCMC_GRID_SIZE	1 // 8
#define ZONE_PLAN_MCMC_BLOCK_SIZE	32 // 128

#define ZONE_GRID_SIZE 10 // 200
#define ZONE_CELL_LEN 400 // 20

struct zone_type {
	int type;
	int level;
};

struct zone_plan {
	zone_type zones[ZONE_GRID_SIZE][ZONE_GRID_SIZE];
	float score;
};

void zonePlanMCMCGPUfunc(zone_plan** bestPlan, int numIterations);

#endif
