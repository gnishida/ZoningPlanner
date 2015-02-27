#include "BrushFire.h"
#include <iterator>

using namespace cv;
using namespace std;

namespace brushfire {

BrushFire::BrushFire(int width, int height, int numStoreTypes, vector<uchar>& zone) : width(width), height(height), numStoreTypes(numStoreTypes) {
	this->zone.resize(width * height);

	copy(zone.begin(), zone.end(), this->zone.begin());
	init();
}

BrushFire& BrushFire::operator=(const BrushFire &ref) {
	width = ref.width;
	height = ref.height;
	numStoreTypes = ref.numStoreTypes;

	zone.resize(width * height);
	copy(ref.zone.begin(), ref.zone.end(), zone.begin());

	dist.resize(numStoreTypes, vector<int>(width * height, 0));
	obst.resize(numStoreTypes, vector<int>(width * height, 0));
	toRaise.resize(numStoreTypes, vector<bool>(width * height, false));
	for (int i = 0; i < numStoreTypes; ++i) {
		copy(ref.dist[i].begin(), ref.dist[i].end(), dist[i].begin());
		copy(ref.obst[i].begin(), ref.obst[i].end(), obst[i].begin());
		copy(ref.toRaise[i].begin(), ref.toRaise[i].end(), toRaise[i].begin());
	}

	// queueはコピーしなくて良いでしょう？

	return *this;
}

/**
 * キューに入った更新情報に基づいて、距離マップを更新する。
 */
void BrushFire::updateDistanceMap() {
	while (!queue.empty()) {
		std::pair<int, int> s = queue.front();
		queue.pop_front();

		int featureId = s.second;

		if (toRaise[featureId][s.first]) {
			raise(s.first, featureId);
		} else if (isOcc(obst[featureId][s.first], featureId)) {
			lower(s.first, featureId);
		}
	}
}

/**
 * 指定されたインデックス番号のセルに、指定された種類の店を追加する。
 *
 * @param s			セルのインデックス番号
 * @param featureId	店の種類
 */
void BrushFire::setStore(int s, int featureId) {
	// put stores
	obst[featureId][s] = s;
	dist[featureId][s] = 0;

	zone[s] = featureId + 1;

	queue.push_back(std::make_pair(s, featureId));
}

/**
 * 指定されたインデックス番号のセルの、指定された種類の店を削除する。
 *
 * @param s			セルのインデックス番号
 * @param featureId	店の種類
 */
void BrushFire::removeStore(int s, int featureId) {
	clearCell(s, featureId);

	toRaise[featureId][s] = true;
	zone[s] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int BrushFire::check() {
	int count = 0;

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			for (int k = 0; k < numStoreTypes; ++k) {
				int min_dist = MAX_DIST;
				for (int r2 = 0; r2 < height; ++r2) {
					for (int c2 = 0; c2 < width; ++c2) {
						if (zone[r2 * width + c2] - 1 == k) {
							int d = distance(r2 * width + c2, r * width + c);
							if (d < min_dist) {
								min_dist = d;
							}
						}
					}
				}

				if (dist[k][r * width + c] != min_dist) {
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

void BrushFire::init() {
	queue.clear();

	dist.resize(numStoreTypes, vector<int>(width * height, 0));
	obst.resize(numStoreTypes, vector<int>(width * height, 0));
	toRaise.resize(numStoreTypes, vector<bool>(width * height, false));

	for (int i = 0; i < width * height; ++i) {
		for (int k = 0; k < numStoreTypes; ++k) {
			toRaise[k][i] = false;
			if (zone[i] - 1 == k) {
				setStore(i, k);
			} else {
				dist[k][i] = MAX_DIST;
				obst[k][i] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap();
}

bool BrushFire::isOcc(int s, int featureId) {
	return obst[featureId][s] == s;
}

int BrushFire::distance(int pos1, int pos2) {
	int x1 = pos1 % width;
	int y1 = pos1 / width;
	int x2 = pos2 % width;
	int y2 = pos2 / width;

	return abs(x1 - x2) + abs(y1 - y2);
}

void BrushFire::clearCell(int s, int featureId) {
	dist[featureId][s] = MAX_DIST;
	obst[featureId][s] = BF_CLEARED;
}

void BrushFire::raise(int s, int featureId) {
	Point2i adj[4];
	adj[0] = Point2i(-1, 0); adj[1] = Point2i(1, 0); adj[2] = Point2i(0, -1); adj[3] = Point2i(0, 1);

	int x = s % width;
	int y = s / width;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
		int n = ny * width + nx;

		if (obst[featureId][n] != BF_CLEARED && !toRaise[featureId][n]) {
			if (!isOcc(obst[featureId][n], featureId)) {
				clearCell(n, featureId);
				toRaise[featureId][n] = true;
			}
			queue.push_back(std::make_pair(n, featureId));
		}
	}

	toRaise[featureId][s] = false;
}

void BrushFire::lower(int s, int featureId) {
	Point2i adj[4];
	adj[0] = Point2i(-1, 0); adj[1] = Point2i(1, 0); adj[2] = Point2i(0, -1); adj[3] = Point2i(0, 1);

	int x = s % width;
	int y = s / width;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
		int n = ny * width + nx;

		if (!toRaise[featureId][n]) {
			int d = distance(obst[featureId][s], n);
			if (d < dist[featureId][n]) {
				dist[featureId][n] = d;
				obst[featureId][n] = obst[featureId][s];
				queue.push_back(std::make_pair(n, featureId));
			}
		}
	}
}

}
