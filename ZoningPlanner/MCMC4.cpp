#include "MCMC4.h"

namespace mcmc4 {

MCMC4::MCMC4(float city_length) {
	this->city_length = city_length;
}

void MCMC4::setPreferences(std::vector<std::vector<float> >& preference) {
	this->preferences = preference;
}

void MCMC4::addPreference(std::vector<float>& preference) {
	this->preferences.push_back(preference);
}

void MCMC4::setPreferenceForLandValue(std::vector<float>& preference_for_land_value) {
	this->preference_for_land_value = preference_for_land_value;
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
void MCMC4::findBestPlan(int** zones, int* city_size, std::vector<float>& zoneTypeDistribution, int start_size, int num_layers, std::vector<std::pair<Polygon2D, ZoneType> >& init_zones) {
	srand(10);
	*city_size = start_size;

	*zones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
	
	// 初期プランを生成
	int* fixed_zones;
	generateFixedZoning(*city_size, init_zones, &fixed_zones);
	generateZoningPlan(*city_size, *zones, zoneTypeDistribution, fixed_zones);

	int max_iterations = 10000;

	for (int layer = 0; layer < num_layers; ++layer) {
		if (layer == 0) {
			optimize(*city_size, max_iterations, fixed_zones, *zones);
		} else {
			optimize2(*city_size, max_iterations, fixed_zones, *zones);
		}
		int* tmpZones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		memcpy(tmpZones, *zones, sizeof(int) * (*city_size) * (*city_size));

		free(*zones);

		// ゾーンマップを、たて、よこ、２倍ずつに増やす
		*city_size *= 2;
		*zones = (int*)malloc(sizeof(int) * (*city_size) * (*city_size));
		free(fixed_zones);
		generateFixedZoning(*city_size, init_zones, &fixed_zones);
		for (int r = 0; r < *city_size; ++r) {
			for (int c = 0; c < *city_size; ++c) {
				int oldR = r / 2;
				int oldC = c / 2;
				if (fixed_zones[r * (*city_size) + c] != ZoneType::TYPE_UNDEFINED) {
					(*zones)[r * (*city_size) + c] = fixed_zones[r * (*city_size) + c];
				} else {
					(*zones)[r * (*city_size) + c] = tmpZones[(int)(oldR * (*city_size) * 0.5 + oldC)];
				}
			}
		}

		max_iterations *= 0.5;

		free(tmpZones);
	}
	
	saveZoneImage(*city_size, *zones, "zone_final.png");
	//saveZone(city_size, zone, "zone_final.txt");
}

void MCMC4::computeDistanceMap(int city_size, int* zones, int** dist) {
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

void MCMC4::saveZoneImage(int city_size, int* zones, char* filename) {
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

void MCMC4::loadZone(int city_size, int* zones, char* filename) {
	FILE* fp = fopen(filename, "r");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fscanf(fp, "%d,", &zones[r * city_size + c]);
		}
	}

	fclose(fp);
}

void MCMC4::saveZone(int city_size, int* zones, char* filename) {
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

void MCMC4::dumpZone(int city_size, int* zones) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%d ", zones[r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void MCMC4::dumpDist(int city_size, int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[(r * city_size + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

float MCMC4::distToFeature(float dist) {
	return exp(-K * dist);
}

std::vector<float> MCMC4::distToFeature(std::vector<float>& dist) {
	std::vector<float> ret(dist.size());
	
	for (int i = 0; i < dist.size(); ++i) {
		ret[i] = distToFeature(dist[i]);
	}

	return ret;
}

float MCMC4::featureToDist(float feature) {
	return -logf(feature) / K;
}

std::vector<float> MCMC4::featureToDist(std::vector<float>& feature) {
	std::vector<float> ret(feature.size());

	for (int i = 0; i < 7; ++i) {
		ret[i] = featureToDist(feature[i]);
	}
	ret[7] = feature[7];

	return ret;
}

float MCMC4::dot(std::vector<float> v1, std::vector<float> v2) {
	float ret = 0.0f;

	for (int i = 0; i < v1.size(); ++i) {
		ret += v1[i] * v2[i];
	}

	return ret;
}







float MCMC4::randf() {
	return (float)rand() / RAND_MAX;
}

float MCMC4::randf(float a, float b) {
	return randf() * (b - a) + a;
}

int MCMC4::sampleFromCdf(float* cdf, int num) {
	float rnd = randf(0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

int MCMC4::sampleFromPdf(float* pdf, int num) {
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

inline bool MCMC4::isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int MCMC4::distance(int city_size, int pos1, int pos2) {
	int x1 = pos1 % city_size;
	int y1 = pos1 / city_size;
	int x2 = pos2 % city_size;
	int y2 = pos2 / city_size;

	return abs(x1 - x2) + abs(y1 - y2);
}

void MCMC4::clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void MCMC4::raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC4::lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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

void MCMC4::updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise) {
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

void MCMC4::setStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

void MCMC4::removeStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s * NUM_FEATURES + featureId] = true;

	queue.push_back(std::make_pair(s, featureId));
}

float MCMC4::min3(float distToStore, float distToAmusement, float distToFactory) {
	return std::min(std::min(distToStore, distToAmusement), distToFactory);
}

/**
 * 特徴量から、価格インデックスを決定する。
 * 価格インデックスは、おおむね 0から1 ぐらいの範囲になる。
 * 実際の価格は、アパートの場合、価格インデックス x 1000、一戸建ての場合、価格インデックス x 300K とする。
 */
float MCMC4::computePriceIndex(std::vector<float>& feature) {
	float s = dot(preference_for_land_value, feature) + 0.3;
	if (s < 0) s = 0;
	return sqrtf(s);
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
void MCMC4::computeFeature(int city_size, int* zones, int* dist, int s, std::vector<float>& feature) {
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

bool MCMC4::GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight) { return rLeft.first > rRight.first; }

/** 
 * ゾーンのスコアを計算する。
 */
float MCMC4::computeScore(int city_size, int* zones, int* dist) {
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
int MCMC4::check(int city_size, int* zones, int* dist) {
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
 * 固定ゾーンのデータを作成する。
 */
void MCMC4::generateFixedZoning(int city_size, std::vector<std::pair<Polygon2D, ZoneType> >& init_zones, int** fixed_zones) {
	*fixed_zones = (int*)malloc(sizeof(int) * city_size * city_size);

	// init_zonesに含まれるセルは、ゾーンをセットする
	int numCells = 0;
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			QVector2D pt = indexToPosition(r * city_size + c, city_size);

			// 除外する
			bool fixed = false;
			int type;
			for (int z = 0; z < init_zones.size(); ++z) {
				if (init_zones[z].first.contains(pt)) {
					fixed = true;
					type = init_zones[z].second.type();
					break;
				}
			}

			if (fixed) {
				(*fixed_zones)[r * city_size + c] = type;
			} else {
				(*fixed_zones)[r * city_size + c] = ZoneType::TYPE_UNDEFINED;
			}
		}
	}
}

/**
 * ゾーンプランを生成する。
 */
void MCMC4::generateZoningPlan(int city_size, int* zones, std::vector<float> zoneTypeDistribution, int* fixed_zones) {
	// init_zonesに含まれないセルをカウントする
	int numCells = 0;
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			if (fixed_zones[r * city_size + c] == ZoneType::TYPE_UNDEFINED) {
				numCells++;
			}
		}
	}

	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = numCells * zoneTypeDistribution[i];
	}

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			// 除隊対象か？
			if (fixed_zones[r * city_size + c] != ZoneType::TYPE_UNDEFINED) {
				zones[r * city_size + c] = fixed_zones[r * city_size + c];
				continue;
			}

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
void MCMC4::optimize(int city_size, int max_iterations, int* fixed_zones, int* bestZone) {
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
			if (fixed_zones[s1] == ZoneType::TYPE_UNDEFINED && zone[s1] > 0) break;
		}
		while (true) {
			s2 = rand() % (city_size * city_size);
			if (fixed_zones[s2] == ZoneType::TYPE_UNDEFINED && zone[s2] == 0) break;
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
	saveZoneImage(city_size, bestZone, filename);
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
void MCMC4::optimize2(int city_size, int max_iterations, int* fixed_zones, int* bestZone) {
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
	saveZoneImage(city_size, bestZone, filename);
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
 * zonesのインデックス番号を座標に変換する。
 */
QVector2D MCMC4::indexToPosition(int index, int city_size) const {
	int cell_len = city_length / city_size;

	int c = index % city_size;
	int r = index / city_size;

	return QVector2D(((float)c + 0.5) * cell_len - city_length * 0.5, ((float)r + 0.5) * cell_len - city_length * 0.5);
}


};