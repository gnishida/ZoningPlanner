#include "ExhaustiveSearch.h"
#include <algorithm>
#include <time.h>
#include "MCMCUtil.h"
#include "BrushFire.h"

namespace exhaustive_search {

void ExhaustiveSearch::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void ExhaustiveSearch::findOptimalPlan(vector<uchar>& zones, vector<float>& zoneTypeDistribution, int city_size) {
	srand(10);

	int numCells = city_size * city_size;
	zones.resize(numCells);

	vector<uchar> best_zones(numCells);
	
	// 各ゾーンタイプの数を計算
	std::vector<int> numRemainings(NUM_FEATURES + 1);
	int actualNumCells = 0;
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = numCells * zoneTypeDistribution[i] + 0.5f;
		actualNumCells += numRemainings[i];
	}

	if (actualNumCells != numCells) {
		numRemainings[0] += numCells - actualNumCells;
	}

	// 初期ゾーンを生成
	int index = 0;
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		for (int j = 0; j < numRemainings[i]; ++j) {
			zones[index++] = i;
		}
	}

	/*
	zones[0] = 0;
	zones[1] = 0;
	zones[2] = 0;
	zones[3] = 0;
	zones[4] = 4;
	zones[5] = 0;
	zones[6] = 0;
	zones[7] = 1;
	zones[8] = 2;
	zones[9] = 0;
	zones[10] = 0;
	zones[11] = 1;
	zones[12] = 0;
	zones[13] = 3;
	zones[14] = 3;
	zones[15] = 5;
	*/

	// 予想される全組合せ数
	unsigned long expected_num = 1;
	{
		int total_num = numCells;
		for (int i = 0; i < numRemainings.size(); ++i) {
			expected_num *= nCk(total_num, numRemainings[i]);
			total_num -= numRemainings[i];
		}
	}

	float best_score = -std::numeric_limits<float>::max();
	unsigned long count = 0;
	QMap<QString, bool> checked_zones;
	clock_t start = clock();
	do {
		// ゾーンタイプを文字列に変換
		QString zones_str;
		for (int i = 0; i < numCells; ++i) {
			zones_str += QString().number(zones[i]);
		}

		if (!checked_zones.contains(zones_str)) {
			float score = computeScore(city_size, zones);
			if (score > best_score) {
				best_score = score;
				copy(zones.begin(), zones.end(), best_zones.begin());
			}
			if (++count % 10000 == 0) {
				clock_t now = clock();
				int sec = (double)(now - start) / CLOCKS_PER_SEC;
				int mi = sec / 60;
				sec = sec - mi * 60;
				printf("searching [count = %ld/%ld (%.2lf %%)] (%d min %d sec)\n", count, expected_num, (float)count / expected_num * 100, mi, sec);
			}
		}
	} while (std::next_permutation(zones.begin(), zones.end()));

	printf("seraching best done. [count = %d], [best_score = %lf]\n", count, best_score);

	copy(best_zones.begin(), best_zones.end(), zones.begin());

	mcmcutil::MCMCUtil::saveZoneImage(city_size, zones, "zone_optimal.png");
}

/**
 * ゾーンプランのスコアを計算する。
 * 当該ゾーンプランに対して、人を最適に配置し、スコアを計算する。
 * 
 * @param city_size		グリッドの一辺のサイズ
 * @param zones			ゾーンプラン配列
 * @return				スコア
 */
float ExhaustiveSearch::computeScore(int city_size, vector<uchar>& zones) {
	brushfire::BrushFire bf(city_size, city_size, NUM_FEATURES, zones);

	float score = mcmcutil::MCMCUtil::computeScore(city_size, NUM_FEATURES, bf.zones(), bf.distMap(), preferences);

	return score;
}

unsigned ExhaustiveSearch::nCk(unsigned n, unsigned k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

}
