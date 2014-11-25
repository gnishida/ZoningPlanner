#include "PeopleAllocation.cuh"

__device__
float dot(float* preference, float* feature) {
	float ret = 0.0;
	for (int i = 0; i < 9; ++i) {
		ret += preference[i] * feature[i];
	}
	return ret;
}

__device__
void swap(float* v1, float* v2) {
	float temp = *v1;
	*v1 = *v2;
	*v2 = temp;
}

__global__
void movePeopleGPUKernel(cuda_person* people, int numPeople) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = gridDim.x * blockDim.x;

	for (int loop = 0; loop < 1; ++loop) {
		for (int i = idx; i < numPeople; i += step) {
			float max_increase = 0.0f;
			int swap_id = -1;
			for (int j = i + step; j < numPeople; j += step) {
				if (people[i].type == people[j].type) continue;

				float increase = dot(people[j].preference, people[i].feature) + dot(people[i].preference, people[j].feature) - people[i].score - people[j].score;
				if (increase > max_increase) {
					max_increase = increase;
					swap_id = j;
				}
			}

			if (swap_id >= 0) {
				swap(&people[i].homeLocation_x, &people[swap_id].homeLocation_x);
				swap(&people[i].homeLocation_y, &people[swap_id].homeLocation_y);
				for (int k = 0; k < 9; ++k) {
					swap(&people[i].feature[k], &people[swap_id].feature[k]);
				}

				people[i].score = dot(people[i].preference, people[i].feature);
				people[swap_id].score = dot(people[swap_id].preference, people[swap_id].feature);
			}
		}
	}
}

/**
 * 人を動かす（CUDA版）
 */
void movePeopleGPUfunc(cuda_person* people, int numPeople) {
	// デバイスメモリを確保
	float* devResults;
	cudaMalloc((void**)&devResults, sizeof(cuda_person) * numPeople);

	// デバイスメモリへ、人のデータを転送
	cudaMemcpy(devResults, people, sizeof(cuda_person) * numPeople, cudaMemcpyHostToDevice);

	// GPU側の関数を呼び出す
    movePeopleGPUKernel<<<PEOPLE_ALLOCATION_GRID_SIZE, PEOPLE_ALLOCATION_BLOCK_SIZE>>>((cuda_person*)devResults, numPeople);

	// 結果をCPU側のバッファへ転送する
    cudaMemcpy(people, devResults, sizeof(cuda_person) * numPeople, cudaMemcpyDeviceToHost);

	// デバイスメモリを開放する
    cudaFree(devResults);
}
