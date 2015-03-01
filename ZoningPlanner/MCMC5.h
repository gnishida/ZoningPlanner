#pragma once

#include <list>
#include <vector>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Polygon2D.h"
#include "ZoneType.h"

using namespace std;
using namespace cv;

namespace mcmc5 {

#define NUM_FEATURES 5
#define K 0.0005

class MCMC5 {
private:
	float city_length;
	std::vector<std::vector<float> > preferences;

public:
	MCMC5(float city_length);

public:
	void setPreferences(std::vector<std::vector<float> >& preference);
	void addPreference(std::vector<float>& preference);
	void findBestPlan(vector<uchar>& zones, int& city_size, const std::vector<float>& zoneTypeDistribution, int start_size, int num_layers, int max_iterations, float upscale_factor);


private:
	void generateZoningPlan(int city_size, const vector<float>& zoneTypeDistribution, vector<uchar>& zones);
	void adjustZoningPlan(int city_size, const vector<float>& zoneTypeDistribution, vector<uchar>& zones);
	bool accept(float current_score, float proposed_score);
	void optimize(int city_size, int max_iterations, vector<uchar>& bestZone);
	void optimize2(int city_size, int max_iterations, vector<uchar>& bestZone);
};

};