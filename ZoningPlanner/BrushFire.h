#pragma once

#include <list>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

namespace brushfire {

#define MAX_DIST 99
#define BF_CLEARED -1

struct BF_QueueElement {
	int pos;
	int type;

	BF_QueueElement(int pos, int type): pos(pos), type(type) {}
};

class BrushFire {
private:
	int width;					// グリッドの横のセルの数
	int height;					// グリッドの縦のセルの数
	int numStoreTypes;			// ゾーンタイプの数
	vector<uchar> zone;			// ゾーンプラン
	vector<vector<int> > dist;	// 距離マップ
	vector<vector<int> > obst;
	vector<vector<bool> > toRaise;
	std::list<std::pair<int, int> > queue;

public:
	BrushFire(int width, int height, int numStoreTypes, vector<uchar>& zone);
	BrushFire& operator=(const BrushFire &ref);
	
	vector<uchar>& zones() { return zone; }
	vector<vector<int> >& distMap() { return dist; }
	void updateDistanceMap();
	void setStore(int s, int featureId);
	void removeStore(int s, int featureId);
	int check();

private:
	void init();
	bool isOcc(int s, int featureId);
	int distance(int pos1, int pos2);
	void clearCell(int s, int featureId);
	void raise(int s, int featureId);
	void lower(int s, int featureId);
};

}