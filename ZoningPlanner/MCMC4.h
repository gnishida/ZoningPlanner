#pragma once

#include <list>
#include <vector>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Polygon2D.h"
#include "ZoneType.h"

namespace mcmc4 {

#define MAX_DIST 99
#define BF_CLEARED -1
#define NUM_FEATURES 5
#define K 0.0005

struct Point2D {
	int x;
	int y;
};

struct BF_QueueElement {
	int pos;
	int type;

	BF_QueueElement(int pos, int type): pos(pos), type(type) {}
};

class MCMC4 {
private:
	std::vector<std::vector<float> > preferences;
	std::vector<float> preference_for_land_value;

public:
	MCMC4();

public:
	void setPreferences(std::vector<std::vector<float> >& preference);
	void addPreference(std::vector<float>& preference);
	void setPreferenceForLandValue(std::vector<float>& preference_for_land_value);
	void findBestPlan(int** zones, int* city_size, std::vector<float>& zoneTypeDistribution, int start_size, int num_layers, std::vector<std::pair<Polygon2D, ZoneType> >& init_zones);
	void computeDistanceMap(int city_size, int* zones, int** dist);
	void showZone(int city_size, int* zones, char* filename);
	void loadZone(int city_size, int* zones, char* filename);
	void saveZone(int city_size, int* zones, char* filename);
	float computePriceIndex(std::vector<float>& feature);
	void computeFeature(int city_size, int* zones, int* dist, int s, std::vector<float>& feature);
	//void computeRawFeature(int city_size, int* zones, int* dist, int s, std::vector<float> feature);
	void dumpZone(int city_size, int* zones);
	void dumpDist(int city_size, int* dist, int featureId);
	static float distToFeature(float dist);
	static std::vector<float> distToFeature(std::vector<float>& dist);
	static float featureToDist(float feature);
	static std::vector<float> featureToDist(std::vector<float>& dist);
	static float priceToFeature(float priceIndex);
	static float dot(std::vector<float> v1, std::vector<float> v2);

private:
	float randf();
	float randf(float a, float b);
	int sampleFromCdf(float* cdf, int num);
	int sampleFromPdf(float* pdf, int num);
	bool isOcc(int* obst, int s, int featureId);
	int distance(int city_size, int pos1, int pos2);
	void clearCell(int* dist, int* obst, int s, int featureId);
	void raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise);
	void setStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void removeStore(std::list<std::pair<int, int> >& queue, int* zones, int* dist, int* obst, bool* toRaise, int s, int featureId);
	float min3(float distToStore, float distToAmusement, float distToFactory);
	static bool GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight);
	float computeScore(int city_size, int* zones, int* dist);
	int check(int city_size, int* zones, int* dist);
	void generateZoningPlan(int city_size, int* zones, std::vector<float> zoneTypeDistribution);
	void optimize(int city_size, int max_iterations, int* bestZone);
	void optimize2(int city_size, int max_iterations, int* bestZone);
};

};