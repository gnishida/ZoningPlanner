#pragma once

#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Polygon2D.h"
#include "ZoneType.h"

using namespace std;

namespace exhaustive_search {

#define NUM_FEATURES 5	// 住宅タイプ以外の、ゾーンタイプの数（商業、工業、公園、公共、アミューズメント）
#define K 0.0005


class ExhaustiveSearch {
private:
	float city_length;
	std::vector<std::vector<float> > preferences;

public:
	ExhaustiveSearch() {}

	void setPreferences(std::vector<std::vector<float> >& preference);
	void findOptimalPlan(vector<uchar>& zones, vector<float>& zoneTypeDistribution, int city_size);
	
	float computeScore(int city_size, vector<uchar>& zones);

	unsigned nCk(unsigned n, unsigned k);
};

}
