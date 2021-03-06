﻿#include "MCMC5.h"
#include "Util.h"
#include "BrushFire.h"
#include "MCMCUtil.h"
#include "Zoning.h"

namespace mcmc5 {

MCMC5::MCMC5(float city_length) {
	this->city_length = city_length;
}

void MCMC5::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void MCMC5::addPreference(std::vector<float>& preference) {
	this->preferences.push_back(preference);
}

/**
 * MCMCで、ベストプランを計算する。
 *
 * @param zones [OUT]			ベストプランが返却される
 * @param city_size	[OUT]		返却されるプランのグリッドの一辺の長さ
 * @param zoneTypeDistribution	各ゾーンタイプの割合
 * @param start_size			初期時のグリッドの一辺の長さ
 * @param num_stages			階層のステージ数
 * @param max_iterations		MCMCのステップ数
 * @param upscale_factor		次のステージに行った時に、どのぐらいMCMCステップ数を増やすか？
 * @param beta					MCMCのacceptance rate式で使用する係数
 */
void MCMC5::findBestPlan(vector<uchar>& zones, int& city_size, const std::vector<float>& zoneTypeDistribution, int start_size, int num_stages, int max_iterations, float upscale_factor, float beta) {
	srand(10);
	city_size = start_size;

	zones.resize(city_size * city_size);
	
	// 初期プランを生成
	int* fixed_zones;
	generateZoningPlan(city_size, zoneTypeDistribution, zones);

	mcmcutil::MCMCUtil::dumpZone(city_size, zones);

	std::vector<float> scores;
	for (int layer = 0; layer < num_stages; ++layer) {
		if (layer == 0) {
			optimize(city_size, max_iterations, zones, scores, beta);
		} else {
			optimize(city_size, max_iterations, zones, scores, beta);
		}

		vector<uchar> tmpZones(city_size * city_size);
		copy(zones.begin(), zones.end(), tmpZones.begin());

		// ゾーンマップを、たて、よこ、２倍ずつに増やす
		city_size *= 2;
		zones.resize(city_size * city_size);

		for (int r = 0; r < city_size; ++r) {
			for (int c = 0; c < city_size; ++c) {
				int oldR = r / 2;
				int oldC = c / 2;
				zones[r * city_size + c] = tmpZones[(int)(oldR * city_size * 0.5 + oldC)];
			}
		}

		max_iterations *= upscale_factor;
	}
	
	mcmcutil::MCMCUtil::saveZoneImage(city_size, zones, "zone_final.png");
	//saveZone(city_size, zone, "zone_final.txt");

	FILE* fp = fopen("zone_mcmc_scores.txt", "w");
	for (int i = 0; i < scores.size(); ++i) {
		fprintf(fp, "%lf\n", scores[i]);
	}
	fclose(fp);
}

/**
 * 指定されたサイズ、指定されたタイプ分布に基づき、ゾーンプランを生成する。
 *
 * @param city_size		グリッドの一辺のサイズ
 * @param zones			生成されたゾーンプランを返却
 * @param zoneTypeDistribution	指定されたタイプ分布
 */
void MCMC5::generateZoningPlan(int city_size, const vector<float>& zoneTypeDistribution, vector<uchar>& zones) {
	std::vector<int> numRemainings(Zoning::NUM_COMPONENTS + 1);
	int numCells = city_size * city_size;
	zones.resize(numCells);

	int actualNumCells = 0;
	for (int i = 0; i < Zoning::NUM_COMPONENTS + 1; ++i) {
		numRemainings[i] = numCells * zoneTypeDistribution[i] + 0.5f;
		actualNumCells += numRemainings[i];
	}

	if (actualNumCells != numCells) {
		numRemainings[0] += numCells - actualNumCells;
	}

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			int type = Util::sampleFromPdf(numRemainings);
			zones[r * city_size + c] = type;
			numRemainings[type] -= 1;
		}
	}
}

/**
 * Metropolis Hastingsアルゴリズムに基づき、提案プランをacceptするかどうか決定する。
 *
 * @param current_score		現在プランのスコア
 * @param proposed_score	提案プランのスコア
 * @param beta				係数
 * @return					acceptならtrue
 */
bool MCMC5::accept(float current_score, float proposed_score, float beta) {
	if (proposed_score > current_score || Util::genRand() < expf(beta * (proposed_score - current_score))) { 
		return true;
	} else {
		return false;
	}
		
}

/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 *
 * @param city_size			都市の一辺の長さ [m]
 * @param max_iterations	MCMCのiteration数
 * @param bestZone [OUT]	最適ゾーニング
 * @param scores [OUT]		最適ゾーニングのスコア
 * @param beta				MCMCのacceptance rate式で使用する係数
 */
void MCMC5::optimize(int city_size, int max_iterations, vector<uchar>& bestZone, std::vector<float>& scores, float beta) {
	time_t start = clock();

	brushfire::BrushFire bf(city_size, city_size, Zoning::NUM_COMPONENTS, bestZone);
	float curScore = mcmcutil::MCMCUtil::computeScore(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);
	//float curScore = mcmcutil::MCMCUtil::computeScoreCUDA(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);
	float bestScore = curScore;

	for (int iter = 0; iter < max_iterations; ++iter) {
		// バックアップ
		brushfire::BrushFire tempBf = bf;

		// ２つのセルのゾーンタイプを交換
		int s1, s2;
		while (true) {
			while (true) {
				s1 = Util::genRand(0, city_size * city_size);

				// s1として、住宅タイプ以外のゾーンを選択
				if (bf.zones()[s1] > 0) break;
			}

			int dir = Util::genRand(0, 4);
			if (dir == 0) {
				s2 = s1 - 1;
			} else if (dir == 1) {
				s2 = s1 + 1;
			} else if (dir == 2) {
				s2 = s1 - city_size;
			} else {
				s2 = s1 + city_size;
			}

			if (s2 < 0 || s2 >= city_size * city_size) continue;

			// s2として、住宅ゾーンを選択
			if (bf.zones()[s2] == 0) break;
		}

		// move a store
		int featureId = bf.zones()[s1] - 1;
		bf.removeStore(s1, featureId);
		bf.setStore(s2, featureId);
		bf.updateDistanceMap();
		
		float proposedScore = mcmcutil::MCMCUtil::computeScore(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);
		//float proposedScore = mcmcutil::MCMCUtil::computeScoreCUDA(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			copy(bf.zones().begin(), bf.zones().end(), bestZone.begin());
		}

		//printf("%lf -> %lf (best: %lf)\n", curScore, proposedScore, bestScore);

		if (accept(curScore, proposedScore, beta)) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			bf = tempBf;
		}

		scores.push_back(bestScore);
	}

	time_t end = clock();
	printf("city_size: %d, best score: %lf, elapsed: %lf\n", city_size, bestScore, (double)(end - start) / CLOCKS_PER_SEC);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	mcmcutil::MCMCUtil::saveZoneImage(city_size, bestZone, filename);
}


/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 * 各ステップでは、隣接セルをランダムに選択し、ゾーンを交換する。
 *
 * @param city_size			都市の一辺の長さ [m]
 * @param max_iterations	MCMCのiteration数
 * @param bestZone [OUT]	最適ゾーニング
 * @param beta				MCMCのacceptance rate式で使用する係数
 */
void MCMC5::optimize2(int city_size, int max_iterations, vector<uchar>& bestZone, float beta) {
	time_t start = clock();

	brushfire::BrushFire bf(city_size, city_size, Zoning::NUM_COMPONENTS, bestZone);
	
	float curScore = mcmcutil::MCMCUtil::computeScore(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);
	float bestScore = curScore;

	std::vector<float> scores;
	int adj[4];
	adj[0] = -1; adj[1] = 1; adj[2] = -city_size; adj[3] = city_size;
	for (int iter = 0; iter < max_iterations; ++iter) {
		// バックアップ
		brushfire::BrushFire tempBf = bf;

		// ２つの隣接セルを選択
		int s1, s2;
		while (true) {
			s1 = rand() % (city_size * city_size);

			int u = rand() % 4;
			s2 = s1 + adj[u];

			if (s2 < 0 || s2 >= city_size * city_size) continue;
			if (bf.zones()[s1] == bf.zones()[s2]) continue;

			int x1 = s1 % city_size;
			int y1 = s1 / city_size;
			int x2 = s2 % city_size;
			int y2 = s2 / city_size;
			if (abs(x1 - x2) + abs(y1 - y2) > 1) continue;

			break;
		}

		// ２つのセルのゾーンタイプを交換
		int f1 = bf.zones()[s1] - 1;
		int f2 = bf.zones()[s2] - 1;
		bf.zones()[s1] = f2 + 1;
		if (f1 >= 0) {
			bf.removeStore(s1, f1);
		}
		if (f2 >= 0) {
			bf.setStore(s1, f2);
		}
		bf.zones()[s2] = f1 + 1;
		if (f2 >= 0) {
			bf.removeStore(s2, f2);
		}
		if (f1 >= 0) {
			bf.setStore(s2, f1);
		}
		bf.updateDistanceMap();
		
		float proposedScore = mcmcutil::MCMCUtil::computeScore(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			copy(bf.zones().begin(), bf.zones().end(), bestZone.begin());
		}

		if (accept(curScore, proposedScore, beta)) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			bf = tempBf;
		}

		scores.push_back(bestScore);
	}

	time_t end = clock();
	printf("city_size: %d, best score: %lf, elapsed: %lf\n", city_size, bestScore, (double)(end - start) / CLOCKS_PER_SEC);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	mcmcutil::MCMCUtil::saveZoneImage(city_size, bestZone, filename);

	sprintf(filename, "zone_%d_scores.txt", city_size);
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < scores.size(); ++i) {
		fprintf(fp, "%lf\n", scores[i]);
	}
	fclose(fp);
}

};