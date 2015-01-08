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
#include "MCMC3.h"
#include "global.h"

UrbanGeometry::UrbanGeometry(MainWindow* mainWin) {
	this->mainWin = mainWin;

	selectedStore = -1;
	selectedSchool = -1;
	selectedRestaurant = -1;
	selectedPark = -1;
	selectedAmusement = -1;
	selectedLibrary = -1;

	zones.city_length = mainWin->glWidget->vboRenderManager.side;
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

void UrbanGeometry::loadInitZones(const QString& filename) {
	zones.loadInitZones(filename);
}

/**
 * ベストのゾーンプランを探す（シングルスレッド版）
 */
void UrbanGeometry::findBestPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preferences) {
	// 各種ゾーンの配分を取得（住宅、商業、工場、公園、アミューズメント、学校・図書館）
	QStringList distribution = G::g["zoning_type_distribution"].toString().split(",");
	std::vector<float> zoneTypeDistribution(distribution.size());
	for (int i = 0; i < distribution.size(); ++i) {
		zoneTypeDistribution[i] = distribution[i].toFloat();
	}

	// 価格を決定するためのpreference vectorを取得
	QStringList pref_for_land_value = G::g["preference_for_land_value"].toString().split(",");
	std::vector<float> preference_for_land_value(pref_for_land_value.size());
	for (int i = 0; i < pref_for_land_value.size(); ++i) {
		preference_for_land_value[i] = pref_for_land_value[i].toFloat();
	}

	mcmc3::MCMC3 mcmc;
	mcmc.setPreferences(preferences);
	mcmc.setPreferenceForLandValue(preference_for_land_value);
	mcmc.findBestPlan(&zones.zones, &zones.zone_size, zoneTypeDistribution, G::getInt("zoning_start_size"), G::getInt("zoning_num_layers"), zones.init_zones);
}

/**
 * 指定されたpreferenceベクトルに対して、ベストの住宅ゾーンを探す。
 * ゾーンプランは既に生成済みである必要がある。
 * ブロックも生成済みである必要がある。
 */
QVector2D UrbanGeometry::findBestPlace(VBORenderManager& renderManager, std::vector<float>& preference) {
	// 価格を決定するためのpreference vectorを取得
	QStringList pref_for_land_value = G::g["preference_for_land_value"].toString().split(",");
	std::vector<float> preference_for_land_value(pref_for_land_value.size());
	for (int i = 0; i < pref_for_land_value.size(); ++i) {
		preference_for_land_value[i] = pref_for_land_value[i].toFloat();
	}

	mcmc3::MCMC3 mcmc;
	mcmc.setPreferenceForLandValue(preference_for_land_value);

	// 距離マップを生成する
	int* dist;
	mcmc.computeDistanceMap(zones.zone_size, zones.zones, &dist);

	int cell_len = renderManager.side / zones.zone_size;

	float best_score = -std::numeric_limits<float>::max();
	QVector2D ret;
	for (int bi = 0; bi < blocks.size(); ++bi) {
		if (blocks[bi].zone.type() != ZoneType::TYPE_RESIDENTIAL) continue;

		BBox3D bbox;
		blocks[bi].sidewalkContour.getBBox3D(bbox.minPt, bbox.maxPt);
		QVector3D pt = bbox.midPt();

		// 当該ブロックのfeatureを取得
		std::vector<float> feature;
		int s = zones.positionToIndex(QVector2D(pt.x(), pt.y()));
		mcmc.computeFeature(zones.zone_size, zones.zones, dist, s, feature);

		float score = mcmc3::MCMC3::dot(feature, preference);

		if (score > best_score) {
			best_score = score;
			ret.setX(pt.x());
			ret.setY(pt.y());
		}
	}

	free(dist);

	return ret;
}

/**
 * 指定された数のcomparison tasksを生成する。
 * 生成済みのゾーンプランから、ランダムに２つのセルを選択し、それぞれのfeatureを取得してcomparison taskとする。
 * ゾーンプランは既に生成済みである必要がある。
 */
std::vector<std::pair<std::vector<float>, std::vector<float> > > UrbanGeometry::generateTasks(int num) {
	// 価格を決定するためのpreference vectorを取得
	QStringList pref_for_land_value = G::g["preference_for_land_value"].toString().split(",");
	std::vector<float> preference_for_land_value(pref_for_land_value.size());
	for (int i = 0; i < pref_for_land_value.size(); ++i) {
		preference_for_land_value[i] = pref_for_land_value[i].toFloat();
	}

	mcmc3::MCMC3 mcmc;
	int* dist;
	mcmc.setPreferenceForLandValue(preference_for_land_value);
	mcmc.computeDistanceMap(zones.zone_size, zones.zones, &dist);

	std::vector<std::vector<float> > features;
	for (int s = 0; s < zones.zone_size * zones.zone_size; ++s) {
		if (zones.zones[s] != ZoneType::TYPE_RESIDENTIAL) continue;

		std::vector<float> feature;
		mcmc.computeFeature(zones.zone_size, zones.zones, dist, s, feature);

		features.push_back(feature);
	}

	free(dist);

	std::vector<std::pair<std::vector<float>, std::vector<float> > > ret;

	for (int iter = 0; iter < num; ++iter) {
		int com1 = Util::genRand(0, 7);
		int com2 = Util::genRand(0, 7);

		int r1 = Util::genRand(0, features.size());
		int r2;
				
		std::vector<float> option1;
		std::vector<float> option2;
		while (true) {
			r2 = Util::genRand(0, features.size());
			if (r2 == r1) continue;

			std::vector<float> f2 = features[r2];
			f2[com1] = features[r1][com1];
			f2[com2] = features[r1][com2];

			f2[7] = mcmc.computePriceIndex(f2);

			option1 = mcmc3::MCMC3::featureToDist(features[r1]);
			option2 = mcmc3::MCMC3::featureToDist(f2);

			// ２つのoptionが近すぎる場合は、棄却
			float len = 0.0f;
			for (int i = 0; i < option1.size(); ++i) {
				len += SQR(option1[i] - option2[i]);
			}
			if (len < 100) continue;

			break;
		}

		ret.push_back(std::make_pair(option1, option2));
	}

	return ret;
}
