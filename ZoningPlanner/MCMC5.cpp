#include "MCMC5.h"
#include "Util.h"
#include "BrushFire.h"
#include "MCMCUtil.h"

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
 * @param num_layers			MCMCの繰り返し数
 * @param init_zones			ユーザ指定のゾーン
 */
void MCMC5::findBestPlan(vector<uchar>& zones, int& city_size, const std::vector<float>& zoneTypeDistribution, int start_size, int num_layers) {
	srand(10);
	city_size = start_size;

	zones.resize(city_size * city_size);
	
	// 初期プランを生成
	int* fixed_zones;
	generateZoningPlan(city_size, zoneTypeDistribution, zones);

	mcmcutil::MCMCUtil::dumpZone(city_size, zones);

	int max_iterations = 20000;

	for (int layer = 0; layer < num_layers; ++layer) {
		if (layer == 0) {
			optimize(city_size, max_iterations, zones);
		} else {
			optimize(city_size, max_iterations, zones);
		}

		vector<uchar> tmpZones(city_size * city_size);
		copy(zones.begin(), zones.end(), back_inserter(tmpZones));

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

		//max_iterations *= 0.5;
	}
	
	mcmcutil::MCMCUtil::saveZoneImage(city_size, zones, "zone_final.png");
	//saveZone(city_size, zone, "zone_final.txt");
}

void MCMC5::computeDistanceMap(int city_size, vector<uchar>& zones, vector<vector<int> >& dist) {

	brushfire::BrushFire bf(city_size, city_size, NUM_FEATURES, zones);

	dist.resize(NUM_FEATURES, vector<int>(city_size * city_size, 0));
	for (int i = 0; i < NUM_FEATURES; ++i) {
		copy(bf.distMap()[i].begin(), bf.distMap()[i].end(), back_inserter(dist[i]));
	}
}

float MCMC5::featureToDist(float feature) {
	return -logf(feature) / K;
}

std::vector<float> MCMC5::featureToDist(std::vector<float>& feature) {
	std::vector<float> ret(feature.size());

	for (int i = 0; i < 7; ++i) {
		ret[i] = featureToDist(feature[i]);
	}
	ret[7] = feature[7];

	return ret;
}

/**
 * 指定されたサイズ、指定されたタイプ分布に基づき、ゾーンプランを生成する。
 *
 * @param city_size		グリッドの一辺のサイズ
 * @param zones			生成されたゾーンプランを返却
 * @param zoneTypeDistribution	指定されたタイプ分布
 */
void MCMC5::generateZoningPlan(int city_size, const vector<float>& zoneTypeDistribution, vector<uchar>& zones) {
	std::vector<int> numRemainings(NUM_FEATURES + 1);
	int numCells = city_size * city_size;
	zones.resize(numCells);

	int actualNumCells = 0;
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
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
 * @return					acceptならtrue
 */
bool MCMC5::accept(float current_score, float proposed_score) {
	const float alpha = 1.0f;
	if (proposed_score > current_score || Util::genRand() < expf(alpha * proposed_score) / expf(alpha * current_score)) { 
		return true;
	} else {
		return false;
	}
		
}

/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 */
void MCMC5::optimize(int city_size, int max_iterations, vector<uchar>& bestZone) {
	mcmcutil::MCMCUtil::dumpZone(city_size, bestZone);

	brushfire::BrushFire bf(city_size, city_size, NUM_FEATURES, bestZone);

	/*for (int i = 0; i < NUM_FEATURES; ++i) {
		mcmcutil::MCMCUtil::dumpDist(city_size, bf.distMap(), i);
	}*/
	
	float curScore = mcmcutil::MCMCUtil::computeScore(city_size, NUM_FEATURES, bf.zones(), bf.distMap(), preferences);
	float bestScore = curScore;

	std::vector<float> scores;
	float beta = 1.0f;
	for (int iter = 0; iter < max_iterations; ++iter) {
		// バックアップ
		brushfire::BrushFire tempBf = bf;

		// ２つのセルのゾーンタイプを交換
		int s1, s2;
		while (true) {
			s1 = Util::genRand(0, city_size * city_size);

			// s1として、住宅タイプ以外のゾーンを選択
			if (bf.zones()[s1] > 0) break;
		}
		while (true) {
			s2 = Util::genRand(0, city_size * city_size);

			// s2として、住宅ゾーンの１つを選択
			if (bf.zones()[s2] == 0) break;
		}

		// move a store
		int featureId = bf.zones()[s1] - 1;
		//zone[s1] = 0;
		bf.removeStore(s1, featureId);
		//zone[s2] = featureId + 1;
		bf.setStore(s2, featureId);
		bf.updateDistanceMap();

		// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		// debug
		if (bf.zones()[0] == 2 || bf.zones()[city_size-1] == 2 || bf.zones()[(city_size - 1) * city_size] == 2 || bf.zones()[city_size * city_size - 1] == 2) {
			// コーナーに工場がある！
			int xxx = 0;
			//dumpZone(city_size, zone);
			//dumpDist(city_size, dist, 4);
		}

		float proposedScore = mcmcutil::MCMCUtil::computeScore(city_size, NUM_FEATURES, bf.zones(), bf.distMap(), preferences);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			copy(bf.zones().begin(), bf.zones().end(), bestZone.begin());
		}

		//printf("%lf -> %lf (best: %lf)\n", curScore, proposedScore, bestScore);

		if (accept(curScore, proposedScore)) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			bf = tempBf;
		}

		scores.push_back(bestScore);
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	mcmcutil::MCMCUtil::saveZoneImage(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);

	sprintf(filename, "zone_%d_scores.txt", city_size);
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < scores.size(); ++i) {
		fprintf(fp, "%lf\n", scores[i]);
	}
	fclose(fp);
}


/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 * 各ステップでは、隣接セルをランダムに選択し、ゾーンを交換する。
 */
/*
void MCMC5::optimize2(int city_size, int max_iterations, int* fixed_zones, int* bestZone) {
	int* zone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);

	memcpy(zone, bestZone, sizeof(int) * city_size * city_size);

	// for backup
	int* tmpZone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* tmpDist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* tmpObst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);

	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zone[i] - 1 == k) {
				setStore(queue, zone, dist, obst, toRaise, i, k);
			} else {
				dist[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);

	//dumpZone(city_size, zone);
	//dumpDist(city_size, dist, 4);
	//check(city_size, zone, dist);

	float curScore = computeScore(city_size, zone, dist);
	float bestScore = curScore;
	memcpy(bestZone, zone, sizeof(int) * city_size * city_size);

	std::vector<float> scores;
	float beta = 1.0f;
	int adj[4];
	adj[0] = -1; adj[1] = 1; adj[2] = -city_size; adj[3] = city_size;
	for (int iter = 0; iter < max_iterations; ++iter) {
		queue.clear();

		// バックアップ
		memcpy(tmpZone, zone, sizeof(int) * city_size * city_size);
		memcpy(tmpDist, dist, sizeof(int) * city_size * city_size * NUM_FEATURES);
		memcpy(tmpObst, obst, sizeof(int) * city_size * city_size * NUM_FEATURES);

		// ２つの隣接セルを選択
		int s1, s2;
		while (true) {
			s1 = rand() % (city_size * city_size);
			if (fixed_zones[s1] != ZoneType::TYPE_UNDEFINED) continue;

			int u = rand() % 4;
			s2 = s1 + adj[u];

			if (s2 < 0 || s2 >= city_size * city_size) continue;
			if (fixed_zones[s2] != ZoneType::TYPE_UNDEFINED) continue;
			if (zone[s1] == zone[s2]) continue;

			int x1 = s1 % city_size;
			int y1 = s1 / city_size;
			int x2 = s2 % city_size;
			int y2 = s2 / city_size;
			if (abs(x1 - x2) + abs(y1 - y2) > 1) continue;

			break;
		}

		// ２つのセルのゾーンタイプを交換
		int f1 = zone[s1] - 1;
		int f2 = zone[s2] - 1;
		zone[s1] = f2 + 1;
		if (f1 >= 0) {
			removeStore(queue, zone, dist, obst, toRaise, s1, f1);
		}
		if (f2 >= 0) {
			setStore(queue, zone, dist, obst, toRaise, s1, f2);
		}
		zone[s2] = f1 + 1;
		if (f2 >= 0) {
			removeStore(queue, zone, dist, obst, toRaise, s2, f2);
		}
		if (f1 >= 0) {
			setStore(queue, zone, dist, obst, toRaise, s2, f1);
		}
		updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);
		
		//dumpZone(city_size, zone);
		//dumpDist(city_size, dist, 4);
		//if (check(city_size, zone, dist) > 0) break;

		float proposedScore = computeScore(city_size, zone, dist);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			memcpy(bestZone, zone, sizeof(int) * city_size * city_size);
		}

		//printf("%lf -> %lf (best: %lf)\n", curScore, proposedScore, bestScore);

		if (accept(curScore, proposedScore)) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			memcpy(zone, tmpZone, sizeof(int) * city_size * city_size);
			memcpy(dist, tmpDist, sizeof(int) * city_size * city_size * NUM_FEATURES);
			memcpy(obst, tmpObst, sizeof(int) * city_size * city_size * NUM_FEATURES);
		}

		scores.push_back(curScore);
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	saveZoneImage(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);

	sprintf(filename, "zone_%d_scores.txt", city_size);
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < scores.size(); ++i) {
		fprintf(fp, "%lf\n", scores[i]);
	}
	fclose(fp);

	free(tmpZone);
	free(tmpDist);
	free(tmpObst);

	free(zone);
	free(dist);
	free(obst);
	free(toRaise);
}
*/

/**
 * zonesのインデックス番号を座標に変換する。
 */
QVector2D MCMC5::indexToPosition(int index, int city_size) const {
	int cell_len = city_length / city_size;

	int c = index % city_size;
	int r = index / city_size;

	return QVector2D(((float)c + 0.5) * cell_len - city_length * 0.5, ((float)r + 0.5) * cell_len - city_length * 0.5);
}


};