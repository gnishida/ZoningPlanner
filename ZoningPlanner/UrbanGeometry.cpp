﻿#include <algorithm>
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
void UrbanGeometry::findBestPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preference) {
	MCMC mcmc(preference);
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

QVector2D UrbanGeometry::findBestPlace(VBORenderManager& renderManager, std::vector<float>& preference) {
	std::vector<std::vector<float> > preferences;
	preferences.push_back(preference);

	MCMC mcmc(preferences);
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
			float feature[8];
			mcmc.computeFeature(zones.zone_size, zones.zones2, dist, s, feature);
			for (int peopleType = 0; peopleType < preference.size(); ++peopleType) {
				score += feature[0] * preference[0]; // 店
				score += feature[1] * preference[1]; // 学校
				score += feature[2] * preference[2]; // レストラン
				score += feature[3] * preference[3]; // 公園
				score += feature[4] * preference[4]; // アミューズメント
				score += feature[5] * preference[5]; // 図書館
				score += feature[6] * preference[6]; // 騒音
				score += feature[7] * preference[7]; // 汚染
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
