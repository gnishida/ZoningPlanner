#include "MCMCUtil.h"
#include "Util.h"

namespace mcmcutil {

bool MCMCUtil::GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight) { return rLeft.first > rRight.first; }

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
		feature[i] = 1.0 / dist[i][s];
	}
}

/** 
 * ゾーンのスコアを計算する。
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
