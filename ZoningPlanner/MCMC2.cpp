#include "MCMC2.h"

namespace mcmc2 {

MCMC2::MCMC2() {
}

void MCMC2::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void MCMC2::addPreference(std::vector<float>& preference) {
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
void MCMC2::findBestPlan(int** zones, int* city_size, std::vector<float>& zoneTypeDistribution, int start_size, int num_layers, std::vector<std::pair<Polygon2D, ZoneType> >& init_zones) {
	srand(10);
	*city_size = start_size;

	*zones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
	
	// 初期プランを生成
	generateZoningPlan(*city_size, *zones, zoneTypeDistribution);

	int max_iterations = 10000;

	for (int layer = 0; layer < num_layers; ++layer) {
		if (layer == 0) {
			optimize(*city_size, max_iterations, *zones);
		} else {
			optimize2(*city_size, max_iterations, *zones);
		}
		int* tmpZones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		memcpy(tmpZones, *zones, sizeof(int) * (*city_size) * (*city_size));

		free(*zones);

		// ゾーンマップを、たて、よこ、２倍ずつに増やす
		*city_size *= 2;
		*zones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		for (int r = 0; r < *city_size; ++r) {
			for (int c = 0; c < *city_size; ++c) {
				int oldR = r / 2;
				int oldC = c / 2;
				(*zones)[r * (*city_size) + c] = tmpZones[(int)(oldR * (*city_size) * 0.5 + oldC)];
			}
		}

		max_iterations *= 0.5;

		free(tmpZones);
	}
	
	//showZone(city_size, zone, "zone_final.png");
	//saveZone(city_size, zone, "zone_final.txt");
}

void MCMC2::computeDistanceMap(int city_size, int* zones, int** dist) {
	*dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);


	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zones[i] - 1 == k) {
				setStore(queue, zones, *dist, obst, toRaise, i, k);
			} else {
				(*dist)[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(city_size, queue, zones, *dist, obst, toRaise);

	free(obst);
	free(toRaise);
}

void MCMC2::showZone(int city_size, int* zones, char* filename) {
	cv::Mat m(city_size, city_size, CV_8UC3);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			cv::Vec3b p;
			if (zones[r * city_size + c] == 0) {
				p = cv::Vec3b(0, 0, 255);
			} else if (zones[r * city_size + c] == 1) {
				p = cv::Vec3b(255, 0, 0);
			} else if (zones[r * city_size + c] == 2) {
				p = cv::Vec3b(64, 64, 64);
			} else if (zones[r * city_size + c] == 3) {
				p = cv::Vec3b(0, 255, 0);
			} else if (zones[r * city_size + c] == 4) {
				p = cv::Vec3b(255, 0, 255);
			} else if (zones[r * city_size + c] == 5) {
				p = cv::Vec3b(0, 255, 255);
			} else {
				p = cv::Vec3b(255, 255, 255);
			}
			m.at<cv::Vec3b>(r, c) = p;
		}
	}

	cv::imwrite(filename, m);
}

void MCMC2::loadZone(int city_size, int* zones, char* filename) {
	FILE* fp = fopen(filename, "r");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fscanf(fp, "%d,", &zones[r * city_size + c]);
		}
	}

	fclose(fp);
}

void MCMC2::saveZone(int city_size, int* zones, char* filename) {
	FILE* fp = fopen(filename, "w");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fprintf(fp, "%d,", zones[r * city_size + c]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	fclose(fp);
}

void MCMC2::dumpZone(int city_size, int* zones) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%d ", zones[r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void MCMC2::dumpDist(int city_size, int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[(r * city_size + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

float MCMC2::distToFeature(float dist) {
	//return exp(-0.001f * dist);
	return exp(-0.0005f * dist);
}

float MCMC2::dot(std::vector<float> v1, std::vector<float> v2) {
	float ret = 0.0f;

	for (int i = 0; i < v1.size(); ++i) {
		ret += v1[i] * v2[i];
	}

	return ret;
}







float MCMC2::randf() {
	return (float)rand() / RAND_MAX;
}

float MCMC2::randf(float a, float b) {
	return randf() * (b - a) + a;
}

int MCMC2::sampleFromCdf(float* cdf, int num) {
	float rnd = randf(0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

int MCMC2::sampleFromPdf(float* pdf, int num) {
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

	return sampleFromCdf(cdf, num);
}

inline bool MCMC2::isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int MCMC2::distance(int city_size, int pos1, int pos2) {
	int x1 = pos1 % city_size;
	int y1 = pos1 / city_size;
	int x2 = pos2 % city_size;
	int y2 = pos2 / city_size;

	return abs(x1 - x2) + abs(y1 - y2);
}

void MCMC2::clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void MCMC2::raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC2::lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC2::updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise) {
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

void MCMC2::setStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

void MCMC2::removeStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s * NUM_FEATURES + featureId] = true;

	queue.push_back(std::make_pair(s, featureId));
}

float MCMC2::min3(float distToStore, float distToAmusement, float distToFactory) {
	return std::min(std::min(distToStore, distToAmusement), distToFactory);
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
void MCMC2::computeFeature(int city_size, int* zones, int* dist, int s, std::vector<float>& feature) {
	int cell_length = 10000 / city_size;

	feature.resize(7);
	feature[0] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // 店
	feature[1] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 学校
	feature[2] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // レストラン
	feature[3] = distToFeature(dist[s * NUM_FEATURES + 2] * cell_length); // 公園
	feature[4] = distToFeature(dist[s * NUM_FEATURES + 3] * cell_length); // アミューズメント
	feature[5] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 図書館
	feature[6] = distToFeature(dist[s * NUM_FEATURES + 1] * cell_length); // 工場
}

void MCMC2::computeRawFeature(int city_size, int* zones, int* dist, int s, float feature[]) {
	int cell_length = 10000 / city_size;

	feature[0] = dist[s * NUM_FEATURES + 0] * cell_length; // 店
	feature[1] = dist[s * NUM_FEATURES + 4] * cell_length; // 学校
	feature[2] = dist[s * NUM_FEATURES + 0] * cell_length; // レストラン
	feature[3] = dist[s * NUM_FEATURES + 2] * cell_length; // 公園
	feature[4] = dist[s * NUM_FEATURES + 3] * cell_length; // アミューズメント
	feature[5] = dist[s * NUM_FEATURES + 4] * cell_length; // 図書館
	feature[6] = dist[s * NUM_FEATURES + 1] * cell_length; // 工場
}

bool MCMC2::GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight) { return rLeft.first > rRight.first; }

/** 
 * ゾーンのスコアを計算する。
 */
float MCMC2::computeScore(int city_size, int* zones, int* dist) {
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
				cell = all_scores[peopleType][pointer[peopleType]].second;
				sc = all_scores[peopleType][pointer[peopleType]].first;
				pointer[peopleType]++;
			} while (used[cell] == 1);

			used[cell] = 1;
			score += sc;
			count--;
		}
	}

	// メモリ解放
	free(used);
	free(pointer);

	return score / num_zones;
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int MCMC2::check(int city_size, int* zones, int* dist) {
	int count = 0;

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = MAX_DIST;
				for (int r2 = 0; r2 < city_size; ++r2) {
					for (int c2 = 0; c2 < city_size; ++c2) {
						if (zones[r2 * city_size + c2] - 1 == k) {
							int d = distance(city_size, r2 * city_size + c2, r * city_size + c);
							if (d < min_dist) {
								min_dist = d;
							}
						}
					}
				}

				if (dist[(r * city_size + c) * NUM_FEATURES + k] != min_dist) {
					if (count == 0) {
						printf("e.g. (%d, %d) featureId = %d\n", c, r, k);
					}
					count++;
				}
			}
		}
	}
	
	if (count > 0) {
		printf("Check results: #error cells = %d\n", count);
	}

	return count;
}

/**
 * ゾーンプランを生成する。
 */
void MCMC2::generateZoningPlan(int city_size, int* zones, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = city_size * city_size * zoneTypeDistribution[i];
	}

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			int type = sampleFromPdf(numRemainings.data(), numRemainings.size());
			zones[r * city_size + c] = type;
			numRemainings[type] -= 1;
		}
	}
}

/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 */
void MCMC2::optimize(int city_size, int max_iterations, int* bestZone) {
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

	float beta = 1.0f;
	for (int iter = 0; iter < max_iterations; ++iter) {
		queue.clear();

		// バックアップ
		memcpy(tmpZone, zone, sizeof(int) * city_size * city_size);
		memcpy(tmpDist, dist, sizeof(int) * city_size * city_size * NUM_FEATURES);
		memcpy(tmpObst, obst, sizeof(int) * city_size * city_size * NUM_FEATURES);

		// ２つのセルのゾーンタイプを交換
		int s1, s2;
		while (true) {
			s1 = rand() % (city_size * city_size);
			if (zone[s1] > 0) break;
		}
		while (true) {
			s2 = rand() % (city_size * city_size);
			if (zone[s2] == 0) break;
		}

		// move a store
		int featureId = zone[s1] - 1;
		zone[s1] = 0;
		removeStore(queue, zone, dist, obst, toRaise, s1, featureId);
		zone[s2] = featureId + 1;
		setStore(queue, zone, dist, obst, toRaise, s2, featureId);
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

		if (proposedScore > curScore || randf() < expf(proposedScore) / expf(curScore)) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			memcpy(zone, tmpZone, sizeof(int) * city_size * city_size);
			memcpy(dist, tmpDist, sizeof(int) * city_size * city_size * NUM_FEATURES);
			memcpy(obst, tmpObst, sizeof(int) * city_size * city_size * NUM_FEATURES);
		}
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	showZone(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);

	free(tmpZone);
	free(tmpDist);
	free(tmpObst);

	free(zone);
	free(dist);
	free(obst);
	free(toRaise);
}


/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 * 各ステップでは、隣接セルをランダムに選択し、ゾーンを交換する。
 */
void MCMC2::optimize2(int city_size, int max_iterations, int* bestZone) {
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
			int u = rand() % 4;
			s2 = s1 + adj[u];

			if (s2 < 0 || s2 >= city_size * city_size) continue;
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

		if (proposedScore > curScore || randf() < proposedScore / curScore) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			memcpy(zone, tmpZone, sizeof(int) * city_size * city_size);
			memcpy(dist, tmpDist, sizeof(int) * city_size * city_size * NUM_FEATURES);
			memcpy(obst, tmpObst, sizeof(int) * city_size * city_size * NUM_FEATURES);
		}
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	showZone(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);

	free(tmpZone);
	free(tmpDist);
	free(tmpObst);

	free(zone);
	free(dist);
	free(obst);
	free(toRaise);
}

};