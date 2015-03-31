#include "MCMCUtil.h"
#include "Util.h"
#include "LPSolver.h"
#include "people_allocation.cuh"

namespace mcmcutil {

bool MCMCUtil::GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight) { return rLeft.first > rRight.first; }

/**
 * ガウス分布を使って、距離を特徴量に変換する。
 *
 * @param city_size	グリッドの一辺の長さ
 * @param distance	距離
 * @return			特徴量
 */
float MCMCUtil::distToFeature(int city_size, float distance) {
	// ガウス分布を使ってみよう
	float K = 1.4f;
	float sigma = (float)city_size * 0.32;

	return K * expf(-distance*distance/2.0/sigma/sigma);
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
void MCMCUtil::computeFeature(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, int s, std::vector<float>& feature) {
	int cell_length = 10000 / city_size;

	feature.resize(num_features);
	for (int i = 0; i < num_features; ++i) {
		// 距離の逆数
		//feature[i] = 1.0 / dist[i][s];

		// ガウス分布を使ってみよう
		float K = 1.4f;
		float sigma = (float)city_size * 0.32;
		feature[i] = distToFeature(city_size, dist[i][s]);
	}
}

/** 
 * 人を最適配置して、ゾーンプランのスコアを計算する。
 * 人の最適配置は、greedyアルゴリズムで実施する。
 */
float MCMCUtil::computeScore(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, vector<vector<float> > preferences) {
	int num_zones = 0;

	// 各ユーザ毎に、全セルのスコアを計算し、スコアでソートする
	std::vector<std::vector<std::pair<float, int> > > all_scores(preferences.size());
	for (int s = 0; s < city_size * city_size; ++s) {
		if (zones[s] != 0) continue;

		num_zones++;

		std::vector<float> feature;
		computeFeature(city_size, num_features, zones, dist, s, feature);

		for (int peopleType = 0; peopleType < preferences.size(); ++peopleType) {
			float score = Util::dot(feature, preferences[peopleType]);

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
 * 人を最適配置して、ゾーンプランのスコアを計算する。
 * 人の最適配置は、greedyアルゴリズムで実施する。
 */
float MCMCUtil::computeScoreCUDA(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, vector<vector<float> > preferences) {
	int num_zones = 0;

	vector<vector<float> > features;
	for (int s = 0; s < city_size * city_size; ++s) {
		if (zones[s] != 0) continue;

		num_zones++;

		std::vector<float> feature;
		computeFeature(city_size, num_features, zones, dist, s, feature);

		features.push_back(feature);
	}

	// CUDAで、各セルの、各ユーザによるスコアを計算する
	float* results;
	allocate_people(preferences, features, &results);

	num_zones = 0;
	std::vector<std::vector<std::pair<float, int> > > all_scores(preferences.size());
	for (int s = 0; s < city_size * city_size; ++s) {
		if (zones[s] != 0) continue;

		for (int peopleType = 0; peopleType < preferences.size(); ++peopleType) {
			float score = results[peopleType * features.size() + num_zones];

			all_scores[peopleType].push_back(std::make_pair(score, s));
		}
		num_zones++;
	}

	// ソート
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

	return 0;
}

/** 
 * 人を最適配置して、ゾーンプランのスコアを計算する。
 * 人の最適配置は、Linear programmingで実施する。
 */
float MCMCUtil::computeScoreLP(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, vector<vector<float> > preferences) {
	int num_zones = 0;

	// 住宅ゾーンのセルの数をカウントする
	std::vector<std::vector<std::pair<float, int> > > all_scores(preferences.size());
	for (int s = 0; s < city_size * city_size; ++s) {
		if (zones[s] == 0) num_zones++;
	}

	LPSolver lp(preferences.size() * num_zones);

	// 制約をセット
	for (int i = 0; i < preferences.size(); ++i) {
		std::vector<int> colno(num_zones);
		std::vector<double> row(num_zones);
		for (int j = 0; j < num_zones; ++j) {
			colno[j] = i * num_zones + j + 1;
			row[j] = 1.0;
		}
		lp.addConstraint(EQ, row, colno, (float)num_zones / (float)preferences.size());
	}
	for (int i = 0; i < num_zones; ++i) {
		std::vector<int> colno(preferences.size());
		std::vector<double> row(preferences.size());
		for (int j = 0; j < preferences.size(); ++j) {
			colno[j] = i + j * num_zones + 1;
			row[j] = 1.0;
		}
		lp.addConstraint(EQ, row, colno, 1.0);
	}

	// Ojbective関数をセット
	{
		int index = 0;
		std::vector<double> row(preferences.size() * num_zones);
		for (int s = 0; s < city_size * city_size; ++s) {
			if (zones[s] != 0) continue;

			std::vector<float> feature;
			computeFeature(city_size, num_features, zones, dist, s, feature);

			for (int p = 0; p < preferences.size(); ++p) {
				float score = Util::dot(feature, preferences[p]);

				row[p * num_zones + index] = score;
			}

			index++;
		}

		lp.setObjective(row);
	}

	// upper boundをセット
	{
		std::vector<double> values(preferences.size() * num_zones, 1.0);
		lp.setUpperBound(values);
	}

	lp.maximize();
	//lp.displaySolution();

	return lp.getObjective() / num_zones;
}

vector<vector<float> > MCMCUtil::readPreferences(const QString& filename) {
	QFile file(filename);
	file.open(QIODevice::ReadOnly);
 
	// preference vectorを読み込む
	std::vector<std::vector<float> > preferences;

	QTextStream in(&file);
	while (true) {
		QString str = in.readLine(0);
		if (str == NULL) break;

		QStringList preference_list = str.split("\t")[1].split(",");
		std::vector<float> preference;
		for (int i = 0; i < preference_list.size(); ++i) {
			preference.push_back(preference_list[i].toFloat());
		}

		// normalize
		Util::normalize(preference);

		preferences.push_back(preference);
	}

	return preferences;
}

/**
 * ファイルからゾーンプランを読み込む。
 * ファイルには、カンマ区切りで、ＮｘＮの形式でゾーンタイプが記録されていること。
 *
 * @filename	file name
 * @return		ゾーンプラン
 */
vector<uchar> MCMCUtil::readZone(const QString& filename) {
	QFile file(filename);
	file.open(QIODevice::ReadOnly);
 
	std::vector<uchar> zones;

	QTextStream in(&file);
	while (true) {
		QString str = in.readLine(0);
		if (str == NULL) break;

		QStringList zone_list = str.split(",");
		for (int i = 0; i < zone_list.size(); ++i) {
			zones.push_back(zone_list[i].toUInt());
		}
	}

	return zones;
}

void MCMCUtil::saveZoneImage(int city_size, vector<uchar>& zones, char* filename) {
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

void MCMCUtil::dumpZone(int city_size, vector<uchar>& zones) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%d ", zones[r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void MCMCUtil::dumpDist(int city_size, vector<vector<int> >& dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[featureId][r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

}
