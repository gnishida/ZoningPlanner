#include <algorithm>
#include "UrbanGeometry.h"
#include <limits>
#include <iostream>
#include <QFile>
#include "common.h"
#include "global.h"
#include "GraphUtil.h"
#include "MainWindow.h"
#include "Util.h"
#include "VBOPm.h"
#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"
#include "Util.h"
#include <numeric>
#include <boost/thread.hpp>   
#include <boost/date_time.hpp>
#include "MCMC.h"

UrbanGeometry::UrbanGeometry(MainWindow* mainWin) {
	this->mainWin = mainWin;

	zones.load("zoning.xml");

	selectedStore = -1;
	selectedSchool = -1;
	selectedRestaurant = -1;
	selectedPark = -1;
	selectedAmusement = -1;
	selectedLibrary = -1;
}

UrbanGeometry::~UrbanGeometry() {
}

void UrbanGeometry::clear() {
	clearGeometry();
}

void UrbanGeometry::clearGeometry() {
	//if (&mainWin->glWidget->vboRenderManager != NULL) delete &mainWin->glWidget->vboRenderManager;

	roads.clear();
}

/**
 * Adapt all geometry objects to &mainWin->glWidget->vboRenderManager.
 */
void UrbanGeometry::adaptToTerrain() {
	roads.adaptToTerrain(&mainWin->glWidget->vboRenderManager);
	for (int i = 0; i < blocks.size(); ++i) {
		blocks[i].adaptToTerrain(&mainWin->glWidget->vboRenderManager);
	}
}

void UrbanGeometry::loadRoads(const QString &filename) {
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) {
		std::cerr << "The file is not accessible: " << filename.toUtf8().constData() << endl;
		throw "The file is not accessible: " + filename;
	}

	roads.clear();
	GraphUtil::loadRoads(roads, filename);

	roads.adaptToTerrain(&mainWin->glWidget->vboRenderManager);
	roads.updateRoadGraph(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::saveRoads(const QString &filename) {
	QFile file(filename);
	if (!file.open(QIODevice::WriteOnly)) {
		std::cerr << "The file is not accessible: " << filename.toUtf8().constData() << endl;
		throw "The file is not accessible: " + filename;
	}

	GraphUtil::saveRoads(roads, filename);
}

void UrbanGeometry::clearRoads() {
	roads.clear();
	roads.updateRoadGraph(mainWin->glWidget->vboRenderManager);

	blocks.clear();
	VBOPm::generateBlockModels(mainWin->glWidget->vboRenderManager, roads, blocks);
	VBOPm::generateParcelModels(mainWin->glWidget->vboRenderManager, blocks);
}

void UrbanGeometry::loadBlocks(const QString& filename) {
	blocks.load(filename);
	VBOPmBlocks::assignZonesToBlocks(zones, blocks);
	VBOPm::generateBlockModels(mainWin->glWidget->vboRenderManager, roads, blocks);
	VBOPm::generateParcelModels(mainWin->glWidget->vboRenderManager, blocks);
}

void UrbanGeometry::saveBlocks(const QString& filename) {
	blocks.save(filename);
}

/**
 * ベストのゾーンプランを探す（シングルスレッド版）
 */
void UrbanGeometry::findBestPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preferences) {
	MCMC mcmc;
	mcmc.setPreferences(preferences);
	mcmc.findBestPlan(&zones.zones2, &zones.zone_size);

	int cell_len = renderManager.side / zones.zone_size;

	zones.zones.resize(zones.zone_size * zones.zone_size);
	for (int r = 0; r < zones.zone_size; ++r) {
		for (int c = 0; c < zones.zone_size; ++c) {
			Polygon2D polygon;
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * cell_len, -renderManager.side * 0.5 + r * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * cell_len, -renderManager.side * 0.5 + r * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * cell_len, -renderManager.side * 0.5 + (r + 1) * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * cell_len, -renderManager.side * 0.5 + (r + 1) * cell_len));
			zones.zones[r * zones.zone_size + c] = std::make_pair(polygon, ZoneType(zones.zones2[r * zones.zone_size + c], 1));
		}
	}
}

/**
 * 指定されたpreferenceベクトルに対して、ベストの住宅ゾーンを探す。
 * ゾーンプランは既に生成済みである必要がある。
 */
QVector2D UrbanGeometry::findBestPlace(VBORenderManager& renderManager, std::vector<float>& preference) {
	MCMC mcmc;
	int* dist;
	mcmc.computeDistanceMap(zones.zone_size, zones.zones2, &dist);

	int cell_len = renderManager.side / zones.zone_size;

	float best_score = 0.0f;
	QVector2D ret;
	for (int r = 0; r < zones.zone_size; ++r) {
		for (int c = 0; c < zones.zone_size; ++c) {
			int s = r * zones.zone_size + c;

			if (zones.zones2[s] == 0) continue;

			float score = 0.0f;
			float feature[7];
			mcmc.computeFeature(zones.zone_size, zones.zones2, dist, s, feature);
			for (int peopleType = 0; peopleType < preference.size(); ++peopleType) {
				score += feature[0] * preference[0]; // 店
				score += feature[1] * preference[1]; // 学校
				score += feature[2] * preference[2]; // レストラン
				score += feature[3] * preference[3]; // 公園
				score += feature[4] * preference[4]; // アミューズメント
				score += feature[5] * preference[5]; // 図書館
				score += feature[6] * preference[6]; // 工場
			}

			if (score > best_score) {
				best_score = score;
				ret.setX(-renderManager.side * 0.5 + (c + 0.5) * cell_len);
				ret.setY(-renderManager.side * 0.5 + (r + 0.5) * cell_len);
			}
		}
	}

	free(dist);

	return ret;
}

/**
 * アンケートを生成し、preferenceベクトルを計算する。
 * ゾーンプランは既に生成済みである必要がある。
 */
std::vector<std::pair<std::vector<float>, std::vector<float> > > UrbanGeometry::generateTasks(int num) {
	MCMC mcmc;
	int* dist;
	mcmc.computeDistanceMap(zones.zone_size, zones.zones2, &dist);

	std::vector<std::vector<float> > features;
	for (int s = 0; s < zones.zone_size * zones.zone_size; ++s) {
		if (zones.zones2[s] == 0) continue;

		float score = 0.0f;
		float feature[7];
		mcmc.computeRawFeature(zones.zone_size, zones.zones2, dist, s, feature);
		//mcmc.computeFeature(zones.zone_size, zones.zones2, dist, s, feature);

		std::vector<float> f;
		for (int i = 0; i < 7; ++i) f.push_back(feature[i]);
		features.push_back(f);
	}

	free(dist);

	const char msg[7][255] = {"Dist to store", "Dist to school", "Dist to restaurant", "Dist to park", "Dist to amusement facility", "Dist to library", "Dist to factory"};
	std::vector<std::pair<std::vector<float>, std::vector<float> > > ret;

	for (int iter = 0; iter < num; ++iter) {
		int r1 = Util::genRand(0, features.size());
		int r2;
		while (true) {
			r2 = Util::genRand(0, features.size());
			if (r2 == r1) continue;

			float len = 0.0f;
			for (int i = 0; i < 7; ++i) {
				len += SQR(features[r1][i] - features[r2][i]);
			}
			if (len < 100) continue;

			break;
		}

		ret.push_back(std::make_pair(features[r1], features[r2]));

		/*
		printf("Q%d:\n", iter);
		printf("  Plan A\n");
		for (int i = 0; i < 7; ++i) {
			if (features[r1][i] == features[r2][i]) continue;
			printf("    %s: %.0lf [m]\n", msg[i], features[r1][i]);
		}
		printf("  Plan B\n");
		for (int i = 0; i < 7; ++i) {
			if (features[r1][i] == features[r2][i]) continue;
			printf("    %s: %.0lf [m]\n", msg[i], features[r2][i]);
		}
		printf("\n");
		*/
	}

	return ret;
}
