#include "people_allocation.cuh"
#include <assert.h>
#include <algorithm>

#define NUM_CELLS	1024		// １回に扱うセルの数

__device__
float dot(float* preference, float* feature, int numComponents) {
	float ret = 0.0;
	for (int i = 0; i < numComponents; ++i) {
		ret += preference[i] * feature[i];
	}
	return ret;
}

/**
 * GPUカーネル関数
 */
__global__
void cudaComputeScoreKernel(int numComponents, int numUsers, float* preferences, int numCells, float* features, float* results) {
	// ユニークなIDを取得
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// 当該ユーザのpreferenceベクトルを取得
	float preference[10];
	for (int i = 0; i < numComponents; ++i) {
		preference[i] = preferences[idx * numComponents + i];
	}

	__shared__ float sFeatures[NUM_CELLS];

	int numIterations = ceil((float)numCells / NUM_CELLS);

	for (int iter = 0; iter < numIterations; ++iter) {
		// featuresをshared memoryへコピーする
		for (int i = idx; i < NUM_CELLS; i += numUsers) {
			for (int k = 0; k < numComponents; ++k) {
				sFeatures[i * numComponents + k] = features[iter * NUM_CELLS + i * numComponents + k];
			}
		}
		__syncthreads();

		// 各セルに対してスコアを計算する
		for (int i = 0; i < NUM_CELLS; ++i) {
			results[iter * NUM_CELLS + idx * numCells + i] = dot(preference, &sFeatures[i * numComponents], numComponents);
		}
	}
}

/**
 * 各ユーザのpreferenceベクトル、セルのpropertyベクトルに基づき、
 * 各ユーザによる各セルのスコアを計算する。
 *
 * @param preferences		各ユーザのpreferenceベクトル
 * @param features			各セルのpropertyベクトル
 */
void allocate_people(vector<vector<float> > preferences, vector<vector<float> > features, float** results) {
	assert(preferences[0].size() == features[0].size());

	int numUsers = preferences.size();
	int numCells = features.size();
	int numComponents = preferences[0].size();
	
	// preferenceベクトル用に、デバイスメモリを確保
	float* devPreferences;
	cudaMalloc((void**)&devPreferences, sizeof(float) * numComponents * numUsers);

	// propertyベクトル用に、デバイスメモリを確保
	float* devFeatures;
	cudaMalloc((void**)&devFeatures, sizeof(float) * numComponents * numCells);

	// 結果格納用に、デバイスメモリを確保
	float* devResults;
	cudaMalloc((void**)&devResults, sizeof(float) * numCells * numUsers);

	// デバイスメモリへ、preferenceベクトルを転送
	vector<float> arrayPreferences(numUsers * numComponents);
	for (int i = 0; i < numUsers; ++i) {
		copy(preferences[i].begin(), preferences[i].end(), arrayPreferences.begin() + i * numComponents);
	}
	cudaMemcpy(devPreferences, arrayPreferences.data(), sizeof(float) * numComponents * numUsers, cudaMemcpyHostToDevice);

	// デバイスメモリへ、propertyベクトルを転送
	vector<float> arrayFeatures(numCells * numComponents);
	for (int i = 0; i < numCells; ++i) {
		copy(features[i].begin(), features[i].end(), arrayFeatures.begin() + i * numComponents);
	}
	cudaMemcpy(devFeatures, arrayFeatures.data(), sizeof(float) * numComponents * numCells, cudaMemcpyHostToDevice);
	
	// GPU側の関数を呼び出す
    cudaComputeScoreKernel<<<1, numUsers>>>(numComponents, numUsers, devPreferences, numCells, devFeatures, devResults);

	// CPU側のメモリを確保
	*results = new float[numCells * numUsers];

	// 結果をCPU側のバッファへ転送する
    cudaMemcpy(*results, devResults, sizeof(float) * numCells * numUsers, cudaMemcpyDeviceToHost);

	// 結果を表示
	for (int i = 0; i < numUsers; ++i) {
		for (int j = 0; j < numCells; ++j) {
			printf("%lf, ", (*results)[i * numCells + j]);
		}
		printf("\n");
	}


	// デバイスメモリを開放する
	cudaFree(devPreferences);
	cudaFree(devFeatures);
    cudaFree(devResults);
}
