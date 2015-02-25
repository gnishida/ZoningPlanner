#include "ExhaustiveSearch.h"
#include <algorithm>
#include <time.h>

namespace exhaustive_search {

void ExhaustiveSearch::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void ExhaustiveSearch::setPreferenceForLandValue(std::vector<float>& preference_for_land_value) {
	this->preference_for_land_value = preference_for_land_value;
}

void ExhaustiveSearch::findOptimalPlan(int** zones, std::vector<float>& zoneTypeDistribution, int city_size) {
	srand(10);

	int numCells = city_size * city_size;
	*zones = (int*)malloc(sizeof(int) * numCells);
	int* best_zones = (int*)malloc(sizeof(int) * numCells);
	
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
			(*zones)[index++] = i;
		}
	}

	// 予想される全組合せ数
	unsigned long expected_num = 1;
	{
		int total_num = numCells;
		for (int i = 0; i < numRemainings.size(); ++i) {
			expected_num *= nCk(total_num, numRemainings[i]);
			total_num -= numRemainings[i];
		}
	}

	float best_score = 0.0f;
	unsigned long count = 0;
	QMap<QString, bool> checked_zones;
	clock_t start = clock();
	do {
		// ゾーンタイプを文字列に変換
		QString zones_str;
		for (int i = 0; i < numCells; ++i) {
			zones_str += QString().number((*zones)[i]);
		}

		if (!checked_zones.contains(zones_str)) {
			float score = computeScore(city_size, *zones);
			if (score > best_score) {
				best_score = score;
				memcpy(best_zones, *zones, sizeof(int) * numCells);
			}
			if (++count % 10000 == 0) {
				clock_t now = clock();
				printf("searching [count = %ld/%ld (%.2lf )] (%lf sec)\n", count, expected_num, (float)count / expected_num * 100, (double)(now - start) / CLOCKS_PER_SEC);
				start = now;
			}
		}
	} while (std::next_permutation(zones, zones + numCells));

	printf("seraching best done. [count = %d], [best_score = %lf]\n", count, best_score);

	memcpy(*zones, best_zones, sizeof(int) * numCells);

	free(best_zones);

	saveZoneImage(city_size, *zones, "zone_final.png");
}

/**
 * 指定されたインデックスのセルのfeatureを計算し、返却する。
 *
 * @param city_size			グリッドの一辺の長さ
 * @param zones				ゾーンを格納した配列
 * @param dist				距離マップを格納した配列
 * @param s					セルのインデックス
 * @param feature [OUT]		計算されたfeature
 */
void ExhaustiveSearch::computeFeature(int city_size, int* zones, int* dist, int s, std::vector<float>& feature) {
	int cell_length = 10000 / city_size;

	feature.resize(8);
	feature[0] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // 店
	feature[1] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 学校
	feature[2] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // レストラン
	feature[3] = distToFeature(dist[s * NUM_FEATURES + 2] * cell_length); // 公園
	feature[4] = distToFeature(dist[s * NUM_FEATURES + 3] * cell_length); // アミューズメント
	feature[5] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 図書館
	feature[6] = distToFeature(dist[s * NUM_FEATURES + 1] * cell_length); // 工場

	feature[7] = computePriceIndex(feature); // 価格
}

void ExhaustiveSearch::saveZoneImage(int city_size, int* zones, char* filename) {
	cv::Mat m(city_size, city_size, CV_8UC3);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			cv::Vec3b p;
			if (zones[r * city_size + c] == 0) {		// 住宅街（赤色）
				p = cv::Vec3b(0, 0, 255);
			} else if (zones[r * city_size + c] == 1) {	// 商業地（青色）
				p = cv::Vec3b(255, 0, 0);
			} else if (zones[r * city_size + c] == 2) {	// 工業地（灰色）
				p = cv::Vec3b(64, 64, 64);
			} else if (zones[r * city_size + c] == 3) {	// 公園（緑色）
				p = cv::Vec3b(0, 255, 0);
			} else if (zones[r * city_size + c] == 4) {	// 歓楽街（黄色）
				p = cv::Vec3b(0, 255, 255);
			} else if (zones[r * city_size + c] == 5) {	// 公共施設（水色）
				p = cv::Vec3b(255, 255, 0);
			} else {									// その他（黒色）
				p = cv::Vec3b(255, 255, 255);
			}
			m.at<cv::Vec3b>(r, c) = p;
		}
	}

	cv::flip(m, m, 0);
	cv::imwrite(filename, m);
}

void ExhaustiveSearch::dumpDist(int city_size, int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[(r * city_size + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * 特徴量から、価格インデックスを決定する。
 * 価格インデックスは、おおむね 0から1 ぐらいの範囲になる。
 * 実際の価格は、アパートの場合、価格インデックス x 1000、一戸建ての場合、価格インデックス x 300K とする。
 */
float ExhaustiveSearch::computePriceIndex(std::vector<float>& feature) {
	float s = dot(preference_for_land_value, feature) + 0.3;
	if (s < 0) s = 0;
	return sqrtf(s);
}

float ExhaustiveSearch::distToFeature(float dist) {
	return exp(-K * dist);
}

std::vector<float> ExhaustiveSearch::distToFeature(std::vector<float>& dist) {
	std::vector<float> ret(dist.size());
	
	for (int i = 0; i < dist.size(); ++i) {
		ret[i] = distToFeature(dist[i]);
	}

	return ret;
}

float ExhaustiveSearch::dot(std::vector<float> v1, std::vector<float> v2) {
	float ret = 0.0f;

	for (int i = 0; i < v1.size(); ++i) {
		ret += v1[i] * v2[i];
	}

	return ret;
}

/**
 * ゾーンプランのスコアを計算する。
 * 当該ゾーンプランに対して、人を最適に配置し、スコアを計算する。
 * 
 * @param city_size		グリッドの一辺のサイズ
 * @param zones			ゾーンプラン配列
 * @return				スコア
 */
float ExhaustiveSearch::computeScore(int city_size, int* zones) {
	int* dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);

	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zones[i] - 1 == k) {
				setStore(queue, zones, dist, obst, toRaise, i, k);
			} else {
				dist[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	//saveZoneImage(city_size, zones, "zone_now.png");

	updateDistanceMap(city_size, queue, zones, dist, obst, toRaise);

	/*
	for (int i = 0; i < NUM_FEATURES; ++i) {
		dumpDist(city_size, dist, i);		
	}
	*/

	float score = computeScore(city_size, zones, dist);

	free(dist);
	free(obst);
	free(toRaise);

	return score;
}

void ExhaustiveSearch::updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise) {
	while (!queue.empty()) {
		std::pair<int, int> s = queue.front();
		queue.pop_front();

		if (toRaise[s.first * NUM_FEATURES + s.second]) {
			raise(city_size, queue, dist, obst, toRaise, s.first, s.second);
		} else if (isOcc(obst, obst[s.first * NUM_FEATURES + s.second], s.second)) {
			lower(city_size, queue, dist, obst, toRaise, s.first, s.second);
		}
	}
}

void ExhaustiveSearch::setStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

inline bool ExhaustiveSearch::isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int ExhaustiveSearch::distance(int city_size, int pos1, int pos2) {
	int x1 = pos1 % city_size;
	int y1 = pos1 / city_size;
	int x2 = pos2 % city_size;
	int y2 = pos2 / city_size;

	return abs(x1 - x2) + abs(y1 - y2);
}

void ExhaustiveSearch::clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void ExhaustiveSearch::raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % city_size;
	int y = s / city_size;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= city_size || ny < 0 || ny >= city_size) continue;
		int n = ny * city_size + nx;

		if (obst[n * NUM_FEATURES + featureId] != BF_CLEARED && !toRaise[n * NUM_FEATURES + featureId]) {
			if (!isOcc(obst, obst[n * NUM_FEATURES + featureId], featureId)) {
				clearCell(dist, obst, n, featureId);
				toRaise[n * NUM_FEATURES + featureId] = true;
			}
			queue.push_back(std::make_pair(n, featureId));
		}
	}

	toRaise[s * NUM_FEATURES + featureId] = false;
}

void ExhaustiveSearch::lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % city_size;
	int y = s / city_size;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= city_size || ny < 0 || ny >= city_size) continue;
		int n = ny * city_size + nx;

		if (!toRaise[n * NUM_FEATURES + featureId]) {
			int d = distance(city_size, obst[s * NUM_FEATURES + featureId], n);
			if (d < dist[n * NUM_FEATURES + featureId]) {
				dist[n * NUM_FEATURES + featureId] = d;
				obst[n * NUM_FEATURES + featureId] = obst[s * NUM_FEATURES + featureId];
				queue.push_back(std::make_pair(n, featureId));
			}
		}
	}
}

bool ExhaustiveSearch::GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight) { return rLeft.first > rRight.first; }

/** 
 * ゾーンのスコアを計算する。
 * Greedyアルゴリズムにより、各ユーザについて、ベストスコアのセルを「そのセルに住んでいる」と見なしてスコアを計算する。
 * もし、そのセルが既に「使用済み」なら、次に高いスコアのセルを使用してスコアを計算する。
 * 全てのユーザについて、スコアの計算が一巡したら、再び、最初のユーザから、同様にしてスコアを計算していく。
 * 使用済みセルがなくなったら、終了。
 *
 * @param city_size		グリッドの一辺のサイズ
 * @param zones			ゾーンタイプの配列
 * @param dist			距離マップ
 * @return				スコア
 */
float ExhaustiveSearch::computeScore(int city_size, int* zones, int* dist) {
	int num_zones = 0;

	// 各ユーザ毎に、全セルのスコアを計算し、スコアでソートする
	std::vector<std::vector<std::pair<float, int> > > all_scores(preferences.size());
	for (int s = 0; s < city_size * city_size; ++s) {
		if (zones[s] != ZoneType::TYPE_RESIDENTIAL) continue;

		num_zones++;

		std::vector<float> feature;
		computeFeature(city_size, zones, dist, s, feature);

		for (int peopleType = 0; peopleType < preferences.size(); ++peopleType) {
			float score = dot(feature, preferences[peopleType]);

			all_scores[peopleType].push_back(std::make_pair(score, s));
		}
	}
	for (int peopleType = 0; peopleType < preferences.size(); ++peopleType) {
		std::sort(all_scores[peopleType].begin(), all_scores[peopleType].end(), GreaterScore);
	}

	// 使用済みチェック用
	int* used = (int*)malloc(sizeof(int) * city_size * city_size);
	memset(used, 0, sizeof(int) * city_size * city_size);

	// ポインタ
	int* pointer = (int*)malloc(sizeof(int) * preferences.size());
	memset(pointer, 0, sizeof(int) * preferences.size());

	float score = 0.0f;
	int count = num_zones;
	while (count > 0) {
		for (int peopleType = 0; peopleType < preferences.size() && count > 0; ++peopleType) {
			int cell;
			float sc;
			do {
				// 当該ユーザ"peopleType"の、次にハイスコアであるセルと、そのスコアを取得
				cell = all_scores[peopleType][pointer[peopleType]].second;
				sc = all_scores[peopleType][pointer[peopleType]].first;

				// 当該ユーザ"peopleType"の、次のハイスコアのポインタを次へ移す
				pointer[peopleType]++;
			} while (used[cell] == 1);

			// このセル"cell"を、当該ユーザが「済んでいる」として、「使用済み」にする
			used[cell] = 1;

			// スコアを更新する
			score += sc;

			// 未使用セルの数を更新する
			count--;
		}
	}

	// メモリ解放
	free(used);
	free(pointer);

	return score / num_zones;
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
