#pragma once

#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QFile>

using namespace std;
using namespace cv;

namespace mcmcutil {

class MCMCUtil {
protected:
	MCMCUtil() {}

public:
	static bool GreaterScore(const std::pair<float, int>& rLeft, const std::pair<float, int>& rRight);
	static float distToFeature(int city_size, float distance);
	static void computeFeature(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, int s, std::vector<float>& feature);
	static float computeScore(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, vector<vector<float> > preferences);
	static float computeScore2(int city_size, int num_features, vector<uchar>& zones, vector<vector<int> >& dist, vector<vector<float> > preferences);
	static vector<vector<float> > readPreferences(const QString& filename);
	static vector<uchar> readZone(const QString& filename);
	static void saveZoneImage(int city_size, vector<uchar>& zones, char* filename);
	static void dumpZone(int city_size, vector<uchar>& zones);
	static void dumpDist(int city_size, vector<vector<int> >& dist, int featureId);
};

}
