#include "ZonePlanMCMC.cuh"
#include <vector>
#include <iostream>

__device__
unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

__device__
float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

__device__
float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

__device__
float sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

__device__
float sampleFromPdf(unsigned int* randx, float* pdf, int num) {
	if (num == 0) return 0;

	float cdf[40];
	cdf[0] = pdf[0];
	for (int i = 1; i < num; ++i) {
		if (cdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(randx, cdf, num);
}

__device__
float sum(float* values, int num) {
	float total = 0.0f;
	for (int i = 0; i < num; ++i) {
		total += values[i];
	}
	return total;
}

__device__
void swapZoneType(zone_type* z1, zone_type* z2) {
	int temp_type = z1->type;
	int temp_level = z1->level;

	z1->type = z2->type;
	z1->level = z2->level;
	z2->type = temp_type;
	z2->level = temp_level;
}

__device__
float dot2(float* preference, float* feature) {
	float ret = 0.0;
	for (int i = 0; i < 9; ++i) {
		ret += preference[i] * feature[i];
	}
	return ret;
}

__device__
float noise(float distToFactory, float distToAmusement, float distToStore) {
	float Km = 800.0 - distToFactory;
	float Ka = 400.0 - distToAmusement;
	float Ks = 200.0 - distToStore;

	return max(max(max(Km, Ka), Ks), 0.0f);
}

__device__
float pollution(float distToFactory) {
	float Km = 800.0 - distToFactory;

	return max(Km, 0.0f);
}

__global__
void zonePlanMCMCGPUKernel(zone_plan* plan, zone_plan* proposal, zone_plan* bestPlan) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// 乱数初期化
	unsigned int randx = idx;

	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001};

	// 人の好みベクトル
	float preference[10][9];
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0.15; preference[0][3] = 0.15; preference[0][4] = 0.3; preference[0][5] = 0; preference[0][6] = 0.1; preference[0][7] = 0.1; preference[0][8] = 0.2;
	preference[1][0] = 0; preference[1][1] = 0; preference[1][2] = 0.15; preference[1][3] = 0; preference[1][4] = 0.55; preference[1][5] = 0; preference[1][6] = 0.2; preference[1][7] = 0.1; preference[1][8] = 0;
	preference[2][0] = 0; preference[2][1] = 0; preference[2][2] = 0.05; preference[2][3] = 0; preference[2][4] = 0; preference[2][5] = 0; preference[2][6] = 0.25; preference[2][7] = 0.1; preference[2][8] = 0.6;
	preference[3][0] = 0.18; preference[3][1] = 0.17; preference[3][2] = 0; preference[3][3] = 0.17; preference[3][4] = 0; preference[3][5] = 0.08; preference[3][6] = 0.2; preference[3][7] = 0.2; preference[3][8] = 0;
	preference[4][0] = 0.3; preference[4][1] = 0; preference[4][2] = 0.3; preference[4][3] = 0.1; preference[4][4] = 0; preference[4][5] = 0; preference[4][6] = 0.1; preference[4][7] = 0.2; preference[4][8] = 0;
	preference[5][0] = 0.05; preference[5][1] = 0; preference[5][2] = 0.1; preference[5][3] = 0.2; preference[5][4] = 0.1; preference[5][5] = 0; preference[5][6] = 0.1; preference[5][7] = 0.15; preference[5][8] = 0.3;
	preference[6][0] = 0.15; preference[6][1] = 0.1; preference[6][2] = 0; preference[6][3] = 0.15; preference[6][4] = 0; preference[6][5] = 0.1; preference[6][6] = 0.1; preference[6][7] = 0.2; preference[6][8] = 0.2;
	preference[7][0] = 0.2; preference[7][1] = 0; preference[7][2] = 0.25; preference[7][3] = 0; preference[7][4] = 0.15; preference[7][5] = 0; preference[7][6] = 0.1; preference[7][7] = 0.1; preference[7][8] = 0.2;
	preference[8][0] = 0.3; preference[8][1] = 0; preference[8][2] = 0.15; preference[8][3] = 0.05; preference[8][4] = 0; preference[8][5] = 0; preference[8][6] = 0.25; preference[8][7] = 0.25; preference[8][8] = 0;
	preference[9][0] = 0.4; preference[9][1] = 0; preference[9][2] = 0.2; preference[9][3] = 0; preference[9][4] = 0; preference[9][5] = 0; preference[9][6] = 0.2; preference[9][7] = 0.2; preference[9][8] = 0;

	// 人の数の比率
	float ratioPeople[10] = {0.06667, 0.06667, 0.06667, 0.21, 0.09, 0.09, 0.09, 0.12, 0.1, 0.1};
	float levelPeople[3] = {1, 5, 10};

	// 初期プランの作成
	{
		float zoneTypeDistribution[18] = {0.2, 0.38, 0.2, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.02, 0, 0, 0.01, 0, 0, 0.01, 0, 0};
		float Z = sum(zoneTypeDistribution, 18);

		float remainedBlockNum[18];
		for (int zi = 0; zi < 18; ++zi) {
			remainedBlockNum[zi] = zoneTypeDistribution[zi] / Z * 200 * 200;
		}

		for (int r = 0; r < 200; ++r) {
			for (int c = 0; c < 200; ++c) {
				int n = sampleFromPdf(&randx, remainedBlockNum, 18);
				plan[idx].zones[r][c].type = n / 3;
				plan[idx].zones[r][c].level = n % 3 + 1;
				remainedBlockNum[n]--;
			}
		}
	}

	float best_score = 0.0;
	float current_score = 0.0f;
	for (int loop = 0; loop < 100; ++loop) {
		// 提案プランの作成
		{
			for (int r = 0; r < 200; ++r) {
				for (int c = 0; c < 200; ++c) {
					proposal[idx].zones[r][c].type = plan[idx].zones[r][c].type;
					proposal[idx].zones[r][c].level = plan[idx].zones[r][c].level;
				}
			}

			int x1 = randf(&randx, 0, 200);
			int y1 = randf(&randx, 0, 200);
			int x2 = randf(&randx, 0, 200);
			int y2 = randf(&randx, 0, 200);

			swapZoneType(&proposal[idx].zones[y1][x1], &proposal[idx].zones[y2][x2]);
		}

		// 提案プランのスコアを計算
		float proposal_score = 0.0;
		float count = 0.0;

		for (int r = 0; r < 200; ++r) {
			for (int c = 0; c < 200; ++c) {
				// 住宅ブロック以外なら、人がいないので、スキップ
				if (plan[idx].zones[r][c].type != 0) continue;

				// 直近の店、レストランまでの距離を計算
				float distToStore = 4000;
				float distToRestaurant = 4000;
				float distToFactory = 4000;
				float distToPark = 4000;
				float distToAmusement = 4000;
				float distToSchool = 4000;
				float distToLibrary = 4000;
				for (int r2 = 0; r2 < 200; ++r2) {
					for (int c2 = 0; c2 < 200; ++c2) {
						float dist = 20 * sqrtf((r - r2) * (r - r2) + (c - c2) * (c - c2));
						if (proposal[idx].zones[r2][c2].type == 1) { // 店・レストラン
							if (dist < distToStore) {
								distToStore = dist;
								distToRestaurant = dist;
							}
						} else if (proposal[idx].zones[r2][c2].type == 2) { // 工場
							if (dist < distToFactory) {
								distToFactory = dist;
							}
						} else if (proposal[idx].zones[r2][c2].type == 3) { // 公園
							if (dist < distToPark) {
								distToPark = dist;
							}
						} else if (proposal[idx].zones[r2][c2].type == 4) { // アミューズメント
							if (dist < distToAmusement) {
								distToAmusement = dist;
							}
						} else if (proposal[idx].zones[r2][c2].type == 5) { // 学校・図書館
							if (dist < distToSchool) {
								distToSchool = dist;
								distToLibrary = dist;
							}
						}
					}
				}

				// 特徴量を計算
				float feature[9];
				feature[0] = __expf(-K[0] * distToStore);
				feature[1] = __expf(-K[1] * distToSchool);
				feature[2] = __expf(-K[2] * distToRestaurant);
				feature[3] = __expf(-K[3] * distToPark);
				feature[4] = __expf(-K[4] * distToAmusement);
				feature[5] = __expf(-K[5] * distToLibrary);
				feature[6] = __expf(-K[5] * noise(distToFactory, distToAmusement, distToStore));
				feature[7] = __expf(-K[5] * pollution(distToFactory));
				feature[8] = 0; // 駅はなし

				// このブロックにおけるスコアを計算
				for (int i = 0; i < 10; ++i) {
					proposal_score += dot2(preference[i], feature) * ratioPeople[i] * levelPeople[proposal[idx].zones[r][c].level - 1];
					count += ratioPeople[i] * levelPeople[proposal[idx].zones[r][c].level - 1];
				}
			}
		}

		proposal_score /= count;

		continue;

		// ベストプランの更新
		if (proposal_score > best_score) {
			best_score = proposal_score;
			for (int r = 0; r < 200; ++r) {
				for (int c = 0; c < 200; ++c) {
					bestPlan[idx].zones[r][c].type = proposal[idx].zones[r][c].type;
					bestPlan[idx].zones[r][c].level = proposal[idx].zones[r][c].level;
				}
			}
			bestPlan[idx].score = best_score;
		}

		// 現在のプランの提案プランを比較し、accept/rejectを決定
		if (proposal_score > current_score || loop % 10 == 0) {
			// accept
			for (int r = 0; r < 200; ++r) {
				for (int c = 0; c < 200; ++c) {
					plan[idx].zones[r][c].type = proposal[idx].zones[r][c].type;
					plan[idx].zones[r][c].level = proposal[idx].zones[r][c].level;
				}
			}

			current_score = proposal_score;
		}
	}
}

/**
 * MCMCでベストプランを探す（CUDA版）
 */
void zonePlanMCMCGPUfunc(zone_plan** bestPlans) {
	// CPU側のメモリを確保
	//zone_plan* plans;
	//plans = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	*bestPlans = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);

	// デバイスメモリを確保
	zone_plan* devPlan;
	cudaMalloc((void**)&devPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	zone_plan* devProposal;
	cudaMalloc((void**)&devProposal, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	zone_plan* devBestPlan;
	cudaMalloc((void**)&devBestPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);

	printf("start GPU kernel.\n");

	// GPU側のカーネル関数を呼び出す
    zonePlanMCMCGPUKernel<<<ZONE_PLAN_MCMC_GRID_SIZE, ZONE_PLAN_MCMC_BLOCK_SIZE>>>(devPlan, devProposal, devBestPlan);

	// 結果をCPU側のバッファへ転送する
    cudaMemcpy(*bestPlans, devBestPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE, cudaMemcpyDeviceToHost);

	// デバイスメモリを開放する
    cudaFree(devPlan);
    cudaFree(devProposal);
    cudaFree(devBestPlan);

	printf("GPU kernel done.\n");

	// ベストプランを返す
	/*for (int i = 0; i < ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE; ++i) {
		bestPlans[i].score = plans[i].score;

		for (int r = 0; r < 200; ++r) {
			for (int c = 0; c < 200; ++c) {
				bestPlans[i].zones[r][c] = plans[i].zones[r][c];
			}
		}
	}*/

	printf("data copy done.\n");
	//free(plans);
}
