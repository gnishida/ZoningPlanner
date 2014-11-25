#ifndef __ZONE_PLAN_MCMC_CUH__

#include <stdio.h>

#define ZONE_PLAN_MCMC_GRID_SIZE	8
#define ZONE_PLAN_MCMC_BLOCK_SIZE	128

struct zone_type {
	int type;
	int level;
};

struct zone_plan {
	zone_type zones[200][200];
	float score;
};

void zonePlanMCMCGPUfunc();

#endif
