#ifndef __PEOPLE_ALLOCATION_CUH__

#include <stdio.h>

#define PEOPLE_ALLOCATION_GRID_SIZE		8
#define PEOPLE_ALLOCATION_BLOCK_SIZE	128

struct cuda_person {
	int type;
	float homeLocation_x;
	float homeLocation_y;
	float preference[9];
	float feature[9];
	float score;
};

void movePeopleGPUfunc(cuda_person* people, int numPeople);

#endif
