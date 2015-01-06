#include "MCMC.h"

MCMC::MCMC() {
}

void MCMC::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void MCMC::addPreference(std::vector<float>& preference) {
	this->preferences.push_back(preference);
}

void MCMC::findBestPlan(int** zone, int* city_size) {
	srand(10);
	*city_size = 5;

	*zone = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
	
	// initialize the zone
	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f; // 住宅
	zoneTypeDistribution[1] = 0.2f; // 商業
	zoneTypeDistribution[2] = 0.1f; // 工場
	zoneTypeDistribution[3] = 0.1f; // 公園
	zoneTypeDistribution[4] = 0.05f; // アミューズメント
	zoneTypeDistribution[5] = 0.05f; // 学校・図書館

	// 初期プランを生成
	generateZoningPlan(*city_size, *zone, zoneTypeDistribution);
	//loadZone(zone, "zone2.txt");

	int max_iterations = 10000;

	for (int layer = 0; layer < NUM_LAYERS; ++layer) {
		if (layer == 0) {
			optimize(*city_size, max_iterations, *zone);
		} else {
			optimize2(*city_size, max_iterations, *zone);
		}
		int* tmpZone = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		memcpy(tmpZone, *zone, sizeof(int) * (*city_size) * (*city_size));

		free(*zone);

		// ゾーンマップを、たて、よこ、２倍ずつに増やす
		*city_size *= 2;
		*zone = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		for (int r = 0; r < *city_size; ++r) {
			for (int c = 0; c < *city_size; ++c) {
				int oldR = r / 2;
				int oldC = c / 2;
				(*zone)[r * (*city_size) + c] = tmpZone[(int)(oldR * (*city_size) * 0.5 + oldC)];
			}
		}

		max_iterations *= 0.5;

		free(tmpZone);
	}
	
	//showZone(city_size, zone, "zone_final.png");
	//saveZone(city_size, zone, "zone_final.txt");
}

void MCMC::computeDistanceMap(int city_size, int* zone, int** dist) {
	*dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);


	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zone[i] - 1 == k) {
				setStore(queue, zone, *dist, obst, toRaise, i, k);
			} else {
				(*dist)[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(city_size, queue, zone, *dist, obst, toRaise);

	free(obst);
	free(toRaise);
}

void MCMC::showZone(int city_size, int* zone, char* filename) {
	cv::Mat m(city_size, city_size, CV_8UC3);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			cv::Vec3b p;
			if (zone[r * city_size + c] == 0) {
				p = cv::Vec3b(0, 0, 255);
			} else if (zone[r * city_size + c] == 1) {
				p = cv::Vec3b(255, 0, 0);
			} else if (zone[r * city_size + c] == 2) {
				p = cv::Vec3b(64, 64, 64);
			} else if (zone[r * city_size + c] == 3) {
				p = cv::Vec3b(0, 255, 0);
			} else if (zone[r * city_size + c] == 4) {
				p = cv::Vec3b(255, 0, 255);
			} else if (zone[r * city_size + c] == 5) {
				p = cv::Vec3b(0, 255, 255);
			} else {
				p = cv::Vec3b(255, 255, 255);
			}
			m.at<cv::Vec3b>(r, c) = p;
		}
	}

	cv::imwrite(filename, m);
}

void MCMC::loadZone(int city_size, int* zone, char* filename) {
	FILE* fp = fopen(filename, "r");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fscanf(fp, "%d,", &zone[r * city_size + c]);
		}
	}

	fclose(fp);
}

void MCMC::saveZone(int city_size, int* zone, char* filename) {
	FILE* fp = fopen(filename, "w");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fprintf(fp, "%d,", zone[r * city_size + c]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	fclose(fp);
}

void MCMC::dumpZone(int city_size, int* zone) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%d ", zone[r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void MCMC::dumpDist(int city_size, int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[(r * city_size + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

float MCMC::distToFeature(float dist) {
	//return exp(-0.001f * dist);
	return exp(-0.0005f * dist);
}









float MCMC::randf() {
	return (float)rand() / RAND_MAX;
}

float MCMC::randf(float a, float b) {
	return randf() * (b - a) + a;
}

int MCMC::sampleFromCdf(float* cdf, int num) {
	float rnd = randf(0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

int MCMC::sampleFromPdf(float* pdf, int num) {
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

inline bool MCMC::isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int MCMC::distance(int city_size, int pos1, int pos2) {
	int x1 = pos1 % city_size;
	int y1 = pos1 / city_size;
	int x2 = pos2 % city_size;
	int y2 = pos2 / city_size;

	return abs(x1 - x2) + abs(y1 - y2);
}

void MCMC::clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void MCMC::raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC::lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC::updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise) {
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

void MCMC::setStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

void MCMC::removeStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s * NUM_FEATURES + featureId] = true;

	queue.push_back(std::make_pair(s, featureId));
}

float MCMC::min3(float distToStore, float distToAmusement, float distToFactory) {
	return std::min(std::min(distToStore, distToAmusement), distToFactory);
}

void MCMC::computeFeature(int city_size, int* zone, int* dist, int s, float feature[]) {
	int cell_length = 10000 / city_size;

	feature[0] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // 店
	feature[1] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 学校
	feature[2] = distToFeature(dist[s * NUM_FEATURES + 0] * cell_length); // レストラン
	feature[3] = distToFeature(dist[s * NUM_FEATURES + 2] * cell_length); // 公園
	feature[4] = distToFeature(dist[s * NUM_FEATURES + 3] * cell_length); // アミューズメント
	feature[5] = distToFeature(dist[s * NUM_FEATURES + 4] * cell_length); // 図書館
	feature[6] = distToFeature(dist[s * NUM_FEATURES + 1] * cell_length); // 工場
}

void MCMC::computeRawFeature(int city_size, int* zone, int* dist, int s, float feature[]) {
	int cell_length = 10000 / city_size;

	feature[0] = dist[s * NUM_FEATURES + 0] * cell_length; // 店
	feature[1] = dist[s * NUM_FEATURES + 4] * cell_length; // 学校
	feature[2] = dist[s * NUM_FEATURES + 0] * cell_length; // レストラン
	feature[3] = dist[s * NUM_FEATURES + 2] * cell_length; // 公園
	feature[4] = dist[s * NUM_FEATURES + 3] * cell_length; // アミューズメント
	feature[5] = dist[s * NUM_FEATURES + 4] * cell_length; // 図書館
	feature[6] = dist[s * NUM_FEATURES + 1] * cell_length; // 工場
}

/** 
 * ゾーンのスコアを計算する。
 */
float MCMC::computeScore(int city_size, int* zone, int* dist) {
	float ratioPeople = 1.0f / preferences.size();

	float score = 0.0f;

	int num_zones = 0;
	for (int i = 0; i < city_size * city_size; ++i) {
		if (zone[i] == 0) continue;

		num_zones++;

		float feature[7];
		computeFeature(city_size, zone, dist, i, feature);
		for (int peopleType = 0; peopleType < preferences.size(); ++peopleType) {
			score += feature[0] * preferences[peopleType][0] * ratioPeople; // 店
			score += feature[1] * preferences[peopleType][1] * ratioPeople; // 学校
			score += feature[2] * preferences[peopleType][2] * ratioPeople; // レストラン
			score += feature[3] * preferences[peopleType][3] * ratioPeople; // 公園
			score += feature[4] * preferences[peopleType][4] * ratioPeople; // アミューズメント
			score += feature[5] * preferences[peopleType][5] * ratioPeople; // 図書館
			score += feature[6] * preferences[peopleType][6] * ratioPeople; // 工場
		}
	}

	return score / num_zones;
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int MCMC::check(int city_size, int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = MAX_DIST;
				for (int r2 = 0; r2 < city_size; ++r2) {
					for (int c2 = 0; c2 < city_size; ++c2) {
						if (zone[r2 * city_size + c2] - 1 == k) {
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
void MCMC::generateZoningPlan(int city_size, int* zone, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = city_size * city_size * zoneTypeDistribution[i];
	}

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			int type = sampleFromPdf(numRemainings.data(), numRemainings.size());
			zone[r * city_size + c] = type;
			numRemainings[type] -= 1;
		}
	}

	return;

	// デバッグ用
	// 工場を一番上に持っていく
	// そうすれば、良いゾーンプランになるはず。。。
	for (int r = 2; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			if (zone[r * city_size + c] != 2) continue;

			bool done = false;
			for (int r2 = 0; r2 < 2 && !done; ++r2) {
				for (int c2 = 0; c2 < city_size && !done; ++c2) {
					if (zone[r2 * city_size + c2] == 2) continue;

					// 交換する
					int type = zone[r2 * city_size + c2];
					zone[r2 * city_size + c2] = zone[r * city_size + c];
					zone[r * city_size + c] = type;
					done = true;
				}
			}
		}
	}
}

/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 */
void MCMC::optimize(int city_size, int max_iterations, int* bestZone) {
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


/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 * 各ステップでは、隣接セルをランダムに選択し、ゾーンを交換する。
 */
void MCMC::optimize2(int city_size, int max_iterations, int* bestZone) {
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
