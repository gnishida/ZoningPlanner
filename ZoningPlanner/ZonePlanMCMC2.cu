/**
 * Nearest neighbor search
 * マップ内に、店、工場などのゾーンがある確率で配備されている時、
 * 住宅ゾーンから直近の店、工場までのマンハッタン距離を計算する。
 *
 * 各店、工場から周辺に再帰的に距離を更新していくので、O(N)で済む。
 * しかも、GPUで並列化することで、さらに計算時間を短縮できる。
 *
 * shared memoryを使用して高速化できるか？
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>
#include "ZonePlanMCMC2.cuh"

__host__ __device__
unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

__host__ __device__
float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

__host__ __device__
float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

__host__ __device__
int sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

__host__ __device__
int sampleFromPdf(unsigned int* randx, float* pdf, int num) {
	if (num == 0) return 0;

	float cdf[40];
	cdf[0] = pdf[0];
	for (int i = 1; i < num; ++i) {
		if (pdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(randx, cdf, num);
}

/**
 * ゾーンプランを生成する。
 */
__host__
void generateZoningPlan(zone_plan& zoningPlan, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = CITY_SIZE * CITY_SIZE * zoneTypeDistribution[i];
	}

	unsigned int randx = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int type = sampleFromPdf(&randx, numRemainings.data(), numRemainings.size());
			zoningPlan.zones[r][c].type = type;
			numRemainings[type] -= 1;
		}
	}
}

/**
 * 直近の店までの距離を計算する（マルチスレッド版、shared memory使用）
 */
__global__
void computeDistanceToStore(zone_plan* zoningPlan, DistanceMap* distanceMap) {
	__shared__ int sDist[(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][NUM_FEATURES];
	__shared__ uint3 sQueue[QUEUE_MAX + 1];
	__shared__ unsigned int queue_begin;
	__shared__ unsigned int queue_end;

	queue_begin = 0;
	queue_end = 0;
	__syncthreads();

	// global memoryからshared memoryへコピー
	int num_strides = (GPU_BLOCK_SIZE * GPU_BLOCK_SIZE * GPU_BLOCK_SCALE * GPU_BLOCK_SCALE + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
	int r0 = blockIdx.y * GPU_BLOCK_SIZE - GPU_BLOCK_SIZE * (GPU_BLOCK_SCALE - 1) * 0.5;
	int c0 = blockIdx.x * GPU_BLOCK_SIZE - GPU_BLOCK_SIZE * (GPU_BLOCK_SCALE - 1) * 0.5;
	for (int i = 0; i < num_strides; ++i) {
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);

		// これ、忘れてた！！
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		int type = 0;
		if (r0 + r1 >= 0 && r0 + r1 < CITY_SIZE && c0 + c1 >= 0 && c0 + c1 < CITY_SIZE) {
			type = zoningPlan->zones[r0 + r1][c0 + c1].type;
		}
		for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
			distanceMap->distances[r0 + r1][c0 + c1][feature_id] = MAX_DIST;
			if (type - 1 == feature_id) {
				sDist[r1][c1][feature_id] = 0;
				unsigned int q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(c1, r1, feature_id);
			} else {
				sDist[r1][c1][feature_id] = MAX_DIST;
			}
		}
	}
	__syncthreads();

	// 距離マップを生成
	unsigned int q_index;
	while ((q_index = atomicInc(&queue_begin, QUEUE_MAX)) < queue_end) {
	//while (queue_begin < queue_end) {
		uint3 pt = sQueue[q_index];
		if (pt.x < 0 || pt.x >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || pt.y < 0 || pt.y >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		int d = sDist[pt.y][pt.x][pt.z];

		if (pt.y > 0) {
			unsigned int old = atomicMin(&sDist[pt.y-1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x, pt.y-1, pt.z);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y+1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x, pt.y+1, pt.z);
			}
		}
		if (pt.x > 0) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x-1][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x-1, pt.y, pt.z);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x+1][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x+1, pt.y, pt.z);
			}
		}
	}

	__syncthreads();

	// global memoryの距離マップへ、コピーする
	for (int i = 0; i < num_strides; ++i) {
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);

		// これ、忘れてた！！
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		if (r0 + r1 >= 0 && r0 + r1 < CITY_SIZE && c0 + c1 >= 0 && c0 + c1 < CITY_SIZE) {
			for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
				atomicMin(&distanceMap->distances[r0 + r1][c0 + c1][feature_id], sDist[r1][c1][feature_id]);
			}
		}
	}
}

/**
 * 直近の店までの距離を計算する（CPU版）
 */
__host__
void computeDistanceToStoreCPU(zone_plan* zoningPLan, DistanceMap* distanceMap) {
	std::list<int3> queue;

	for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
		for (int cell_id = 0; cell_id < CITY_SIZE * CITY_SIZE; ++cell_id) {
			int r = cell_id / CITY_SIZE;
			int c = cell_id % CITY_SIZE;

			if (zoningPLan->zones[r][c].type - 1 == feature_id) {
				queue.push_back(make_int3(c, r, feature_id));
				distanceMap->distances[r][c][feature_id] = 0;
			} else {
				distanceMap->distances[r][c][feature_id] = MAX_DIST;
			}
		}
	}

	while (!queue.empty()) {
		int3 pt = queue.front();
		queue.pop_front();

		int d = distanceMap->distances[pt.y][pt.x][pt.z];

		if (pt.y > 0) {
			if (distanceMap->distances[pt.y-1][pt.x][pt.z] > d + 1) {
				distanceMap->distances[pt.y-1][pt.x][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x, pt.y-1, pt.z));
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			if (distanceMap->distances[pt.y+1][pt.x][pt.z] > d + 1) {
				distanceMap->distances[pt.y+1][pt.x][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x, pt.y+1, pt.z));
			}
		}
		if (pt.x > 0) {
			if (distanceMap->distances[pt.y][pt.x-1][pt.z] > d + 1) {
				distanceMap->distances[pt.y][pt.x-1][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x-1, pt.y, pt.z));
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			if (distanceMap->distances[pt.y][pt.x+1][pt.z] > d + 1) {
				distanceMap->distances[pt.y][pt.x+1][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x+1, pt.y, pt.z));
			}
		}
	}
}

/**
 * デバッグ用に、ゾーンプランを表示する。
 */
__host__
void showDevZoningPlan(zone_plan* zoningPlan) {
	zone_plan plan;

	CUDA_CALL(cudaMemcpy(&plan, zoningPlan, sizeof(zone_plan), cudaMemcpyDeviceToHost));
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", plan.zones[r][c].type);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * デバッグ用に、ゾーンプランを表示する。
 */
__host__
void showDevDistMap(DistanceMap* distMap, int feature_id) {
	DistanceMap map;

	CUDA_CALL(cudaMemcpy(&map, distMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost));
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", map.distances[r][c][feature_id]);
		}
		printf("\n");
	}
	printf("\n");
}

__host__
void zonePlanMCMCGPUfunc2(zone_plan** hostZoningPlan, int numIterations)
{
	time_t start, end;


	*hostZoningPlan = (zone_plan*)malloc(sizeof(zone_plan));
	DistanceMap* hostDistanceMap = (DistanceMap*)malloc(sizeof(DistanceMap));
	DistanceMap* hostDistanceMap2 = (DistanceMap*)malloc(sizeof(DistanceMap));

	// 距離を初期化
	//memset(hostDistanceMap, MAX_DIST, sizeof(DistanceMap));
	//memset(hostDistanceMap2, MAX_DIST, sizeof(DistanceMap));

	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f;
	zoneTypeDistribution[1] = 0.2f;
	zoneTypeDistribution[2] = 0.1f;
	zoneTypeDistribution[3] = 0.1f;
	zoneTypeDistribution[4] = 0.05f;
	zoneTypeDistribution[5] = 0.05f;
	
	// 初期プランを生成
	start = clock();
	generateZoningPlan(**hostZoningPlan, zoneTypeDistribution);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	
	// 初期プランをデバイスバッファへコピー
	zone_plan* devZoningPlan;
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(zone_plan)));
	CUDA_CALL(cudaMemcpy(devZoningPlan, *hostZoningPlan, sizeof(zone_plan), cudaMemcpyHostToDevice));

	// デバッグ用に、初期プランを表示
	if (CITY_SIZE <= 100) {
		showDevZoningPlan(devZoningPlan);
	}

	// 距離マップ用に、デバイスバッファを確保
	DistanceMap* devDistanceMap;
	CUDA_CALL(cudaMalloc((void**)&devDistanceMap, sizeof(DistanceMap)));

	///////////////////////////////////////////////////////////////////////
	// CPU版で、直近の店までの距離を計算
	/*
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		computeDistanceToStoreCPU(hostZoningPlan, hostDistanceMap2);
	}
	end = clock();
	printf("computeDistanceToStore CPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	*/

	int* devQueueStart;
	CUDA_CALL(cudaMalloc((void**)&devQueueStart, sizeof(int)));
	int* devQueueEnd;
	CUDA_CALL(cudaMalloc((void**)&devQueueEnd, sizeof(int)));

	printf("start...\n");

	///////////////////////////////////////////////////////////////////////
	// warmp up
	computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);

	// マルチスレッドで、直近の店までの距離を計算
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);
		cudaDeviceSynchronize();
	}
	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);



	CUDA_CALL(cudaMemcpy(devZoningPlan, *hostZoningPlan, sizeof(zone_plan), cudaMemcpyHostToDevice));
	showDevZoningPlan(devZoningPlan);
	computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);
	cudaDeviceSynchronize();



	// 距離マップを表示
	showDevDistMap(devDistanceMap, 1);



	// release device buffer
	cudaFree(devZoningPlan);
	cudaFree(devDistanceMap);

	// release host buffer
	//free(hostZoningPlan);
	free(hostDistanceMap);
	free(hostDistanceMap2);

	//cudaDeviceReset();
}
