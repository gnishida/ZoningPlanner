#pragma once

#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Polygon2D.h"
#include "ZoneType.h"

namespace exhaustive_search {

#define MAX_DIST 99
#define BF_CLEARED -1
#define NUM_FEATURES 5	// 住宅タイプ以外の、ゾーンタイプの数（商業、工業、公園、公共、アミューズメント）
#define K 0.0005

struct Point2D {
	int x;
	int y;
};

class ExhaustiveSearch {
private:
	float city_length;
	std::vector<std::vector<float> > preferences;
	std::vector<float> preference_for_land_value;

public:
	ExhaustiveSearch() {}

	void setPreferences(std::vector<std::vector<float> >& preference);
	void setPreferenceForLandValue(std::vector<float>& preference_for_land_value);
	void findOptimalPlan(int** zones, std::vector<float>& zoneTypeDistribution, int city_size);
	void saveZoneImage(int city_size, int* zones, char* filename);
	void dumpDist(int city_size, int* dist, int featureId);
	float computePriceIndex(std::vector<float>& feature);
	float computePriceIndex2(int city_size, int s);
	void computeFeature(int city_size, int* zones, int* dist, int s, std::vector<float>& feature);
	static float distToFeature(float dist);
	static std::vector<float> distToFeature(std::vector<float>& dist);
	static float dot(std::vector<float> v1, std::vector<float> v2);

	float computeScore(int city_size, int* zones);
	void updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise);
	void setStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId);
	bool isOcc(int* obst, int s, int featureId);
	int distance(int city_size, int pos1, int pos2);
	void clearCell(int* dist, int* obst, int s, int featureId);
	void raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	static bool GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight);
	float computeScore(int city_size, int* zones, int* dist);
	unsigned nCk(unsigned n, unsigned k);
};

}
