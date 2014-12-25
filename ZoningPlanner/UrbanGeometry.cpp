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
void UrbanGeometry::findBestPlan(VBORenderManager& renderManager) {
	MCMC mcmc;
	int* plan;
	int size;
	mcmc.findBestPlan(&plan, &size);

	int cell_len = renderManager.side / size;

	zones.zones.resize(size * size);
	for (int r = 0; r < size; ++r) {
		for (int c = 0; c < size; ++c) {
			Polygon2D polygon;
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * cell_len, -renderManager.side * 0.5 + r * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * cell_len, -renderManager.side * 0.5 + r * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * cell_len, -renderManager.side * 0.5 + (r + 1) * cell_len));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * cell_len, -renderManager.side * 0.5 + (r + 1) * cell_len));
			zones.zones[r * size + c] = std::make_pair(polygon, ZoneType(plan[r * size + c], 1));
		}
	}
}

