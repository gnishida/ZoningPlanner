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
#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"
#include "Util.h"
#include <numeric>
#include <boost/thread.hpp>   
#include <boost/date_time.hpp>
#include "global.h"
#include "RoadMeshGenerator.h"
#include "BlockMeshGenerator.h"
#include "VBOVegetation.h"
#include "VBOPmBuildings.h"
#include "BuildingMeshGenerator.h"
#include "ZoneMeshGenerator.h"
#include "ExhaustiveSearch.h"
#include "MCMC5.h"
#include "MCMCUtil.h"
#include "BrushFire.h"

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
	roads.clear();
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::loadRoads(const QString &filename) {
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) {
		std::cerr << "The file is not accessible: " << filename.toUtf8().constData() << endl;
		throw "The file is not accessible: " + filename;
	}

	roads.clear();
	GraphUtil::loadRoads(roads, filename);
	update(mainWin->glWidget->vboRenderManager);
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
	blocks.clear();
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::loadInitZones(const QString& filename) {
	zones.loadInitZones(filename);
}

void UrbanGeometry::generateBlocks() {
	VBOPmBlocks::generateBlocks(zones, roads, blocks);
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::generateParcels() {
	VBOPmParcels::generateParcels(zones, mainWin->glWidget->vboRenderManager, blocks);
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::generateBuildings() {
	VBOPmBuildings::generateBuildings(mainWin->glWidget->vboRenderManager, blocks);
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::generateVegetation() {
	VBOVegetation::generateVegetation(mainWin->glWidget->vboRenderManager, blocks);
	update(mainWin->glWidget->vboRenderManager);
}

void UrbanGeometry::generateAll() {
	VBOPmBlocks::generateBlocks(zones, roads, blocks);
	VBOPmParcels::generateParcels(zones, mainWin->glWidget->vboRenderManager, blocks);
	VBOPmBuildings::generateBuildings(mainWin->glWidget->vboRenderManager, blocks);
	update(mainWin->glWidget->vboRenderManager);
}

/**
 * 道路、歩道、区画、ビル、木のジオミトリを作成しなおす。
 * この関数を頻繁に呼ぶべきではない。道路が更新/生成された時、PMメニューから新規にジオミトリを生成した時だけ。
 */
void UrbanGeometry::update(VBORenderManager& vboRenderManager) {
	// 地面が変わっている可能性などがあるので、ビルなどのジオミトリも一旦削除してしまう。
	// 道路以外のジオミトリは、別途、PMメニューから作成すること
	vboRenderManager.removeStaticGeometry("3d_blocks");
	vboRenderManager.removeStaticGeometry("3d_parks");
	vboRenderManager.removeStaticGeometry("3d_parcels");
	vboRenderManager.removeStaticGeometry("3d_roads");
	vboRenderManager.removeStaticGeometry("3d_roads_inter");
	vboRenderManager.removeStaticGeometry("3d_roads_interCom");
	vboRenderManager.removeStaticGeometry("3d_building");
	vboRenderManager.removeStaticGeometry("3d_building_fac");
	vboRenderManager.removeStaticGeometry("tree");
	vboRenderManager.removeStaticGeometry("streetLamp");
	vboRenderManager.removeStaticGeometry("zoning");

	RoadMeshGenerator::generateRoadMesh(vboRenderManager, roads);
	BlockMeshGenerator::generateBlockMesh(vboRenderManager, blocks);
	BlockMeshGenerator::generateParcelMesh(vboRenderManager, blocks);
	BuildingMeshGenerator::generateBuildingMesh(vboRenderManager, blocks, zones);
	VBOVegetation::generateVegetation(vboRenderManager, blocks);
	ZoneMeshGenerator::generateZoneMesh(vboRenderManager, blocks);
}

/**
 * ベストのゾーンプランを探す（シングルスレッド版）
 * @param renderManager
 * @param preferences			ユーザのpreferenceベクトル配列
 * @param zoning_start_start	初期グリッドの一辺の長さ
 * @param zoning_num_stages		階層のステージ数
 * @param mcmc_steps			MCMCのステップ数
 * @param upscale_factor		次のステージに行った時に、どのぐらいMCMCステップ数を増やすか？
 */
void UrbanGeometry::findBestPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preferences, int zoning_start_size, int zoning_num_stages, int mcmc_steps, float upscale_factor) {
	// 各種ゾーンの配分を取得（住宅、商業、工場、公園、アミューズメント、学校・図書館）
	QStringList distribution = G::g["zoning_type_distribution"].toString().split(",");
	std::vector<float> zoneTypeDistribution(distribution.size());
	for (int i = 0; i < distribution.size(); ++i) {
		zoneTypeDistribution[i] = distribution[i].toFloat();
	}

	mcmc5::MCMC5 mcmc(renderManager.side);
	mcmc.setPreferences(preferences);

	std::vector<uchar> zones;
	int zone_size;
	mcmc.findBestPlan(zones, zone_size, zoneTypeDistribution, zoning_start_size, zoning_num_stages, mcmc_steps, upscale_factor);
}

/**
 * 指定されたpreferenceベクトルに対して、ベストの住宅ゾーンを探す。
 * ゾーンプランは既に生成済みである必要がある。
 * ブロックも生成済みである必要がある。
 */
QVector2D UrbanGeometry::findBestPlace(VBORenderManager& renderManager, std::vector<float>& preference) {
	mcmc5::MCMC5 mcmc(renderManager.side);

	// 距離マップを生成する
	brushfire::BrushFire bf(zones.zone_size, zones.zone_size, Zoning::NUM_COMPONENTS, zones.zones);

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
		mcmcutil::MCMCUtil::computeFeature(zones.zone_size, Zoning::NUM_COMPONENTS, zones.zones, bf.distMap(), s, feature);

		float score = Util::dot(feature, preference);

		if (score > best_score) {
			best_score = score;
			ret.setX(pt.x());
			ret.setY(pt.y());
		}
	}

	return ret;
}

/**
 * 指定された数のcomparison tasksを生成する。
 * 生成済みのゾーンプランから、ランダムに２つのセルを選択し、それぞれのfeatureを取得してcomparison taskとする。
 * ゾーンプランは既に生成済みである必要がある。
 */
std::vector<std::pair<std::vector<float>, std::vector<float> > > UrbanGeometry::generateTasks(VBORenderManager& renderManager, int num) {
	// 価格を決定するためのpreference vectorを取得
	QStringList pref_for_land_value = G::g["preference_for_land_value"].toString().split(",");
	std::vector<float> preference_for_land_value(pref_for_land_value.size());
	for (int i = 0; i < pref_for_land_value.size(); ++i) {
		preference_for_land_value[i] = pref_for_land_value[i].toFloat();
	}

	brushfire::BrushFire bf(zones.zone_size, zones.zone_size, Zoning::NUM_COMPONENTS, zones.zones);
	mcmc5::MCMC5 mcmc(renderManager.side);

	std::vector<std::vector<float> > features;
	for (int s = 0; s < zones.zone_size * zones.zone_size; ++s) {
		if (zones.zones[s] != ZoneType::TYPE_RESIDENTIAL) continue;

		std::vector<float> feature;
		mcmcutil::MCMCUtil::computeFeature(zones.zone_size, Zoning::NUM_COMPONENTS, zones.zones, bf.distMap(), s, feature);

		features.push_back(feature);
	}

	std::vector<std::pair<std::vector<float>, std::vector<float> > > ret;

	for (int iter = 0; iter < num; ++iter) {
		int com1 = Util::genRand(0, Zoning::NUM_COMPONENTS);
		int com2 = Util::genRand(0, Zoning::NUM_COMPONENTS);

		int r1 = Util::genRand(0, zones.zone_size * zones.zone_size);
		int r2;

		std::vector<float> option1;
		std::vector<float> option2;

		for (int k = 0; k < Zoning::NUM_COMPONENTS; ++k) {
			option1[k] = bf.distMap()[k][r1];
		}

		while (true) {
			r2 = Util::genRand(0, zones.zone_size * zones.zone_size);
			if (r2 == r1) continue;

			for (int k = 0; k < Zoning::NUM_COMPONENTS; ++k) {
				option2[k] = bf.distMap()[k][r1];
			}

			option2[com1] = option1[com1];
			option2[com2] = option1[com2];

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

void UrbanGeometry::findOptimalPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preference, int zoning_start_size) {
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

	exhaustive_search::ExhaustiveSearch es;
	es.setPreferences(preference);

	vector<uchar> zones;
	es.findOptimalPlan(zones, zoneTypeDistribution, zoning_start_size);
	//zones.zone_size = zoning_start_size;
}
