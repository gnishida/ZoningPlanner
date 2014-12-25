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

UrbanGeometry::UrbanGeometry(MainWindow* mainWin) {
	this->mainWin = mainWin;

	zones.load("zoning.xml");

	selectedPerson = -1;
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
	/*
	zone_plan plan;
	zone_plan proposal;
	zone_plan bestPlan;
	MCMCstep(numIterations, 0, &plan, &proposal, &bestPlan);

	this->zones.zones.resize(ZONE_GRID_SIZE * ZONE_GRID_SIZE);
	for (int r = 0; r < ZONE_GRID_SIZE; ++r) {
		for (int c = 0; c < ZONE_GRID_SIZE; ++c) {
			Polygon2D polygon;
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * ZONE_CELL_LEN, -renderManager.side * 0.5 + r * ZONE_CELL_LEN));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * ZONE_CELL_LEN, -renderManager.side * 0.5 + r * ZONE_CELL_LEN));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + (c + 1) * ZONE_CELL_LEN, -renderManager.side * 0.5 + (r + 1) * ZONE_CELL_LEN));
			polygon.push_back(QVector2D(-renderManager.side * 0.5 + c * ZONE_CELL_LEN, -renderManager.side * 0.5 + (r + 1) * ZONE_CELL_LEN));
			zones.zones[r * ZONE_GRID_SIZE + c] = std::make_pair(polygon, ZoneType(bestPlan.zones[r][c].type, bestPlan.zones[r][c].level));
		}
	}

	printf("writing zoning to a file (score: %lf)\n", bestPlan.score);
	QString filename = QString("zoning/score_%1.xml").arg(bestPlan.score, 4, 'f', 6);
	zones.save(filename);
	*/
}

/**
 * 住民、オフィス、レストラン、図書館、公園、工場、などなどを配備する
 */
void UrbanGeometry::allocateAll() {
	people.clear();
	offices.clear();
	schools.clear();
	stores.clear();
	restaurants.clear();
	amusements.clear();
	parks.clear();
	libraries.clear();
	factories.clear();
	stations.clear();

	// 予想される数を先に計算する
	std::vector<float> numPeople(4, 0.0f);
	std::vector<float> numCommercials(3, 0.0f);
	std::vector<float> numManufacturings(2, 0.0f);
	std::vector<float> numAmusements(3, 0.0f);
	std::vector<float> numPublics(2, 0.0f);
	
	for (int i = 0; i < blocks.size(); ++i) {
		QVector2D location = QVector2D(blocks.at(i).blockContour.getCentroid());

		// 予測される区画数を計算
		int numParcels = blocks[i].blockContour.area() / blocks[i].zone.parcel_area_mean;

		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
		} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			/*
			// 住人の数を決定
			int num = numParcels * Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.01f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.02f;
			}
			numPeople[0] += num * 0.2f;
			numPeople[1] += num * 0.3f;
			numPeople[2] += num * 0.3f;
			numPeople[3] += num * 0.2f;
			*/
		} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {
			numCommercials[0] += numParcels * 0.6f; // office
			numCommercials[1] += numParcels * 0.2f; // store
			numCommercials[2] += numParcels * 0.2f; // restaurant
		} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {
			numManufacturings[0] += numParcels * 0.2f;
			numManufacturings[1] += numParcels * 0.8f;
		} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {
			numAmusements[0] += numParcels * 0.6f; // amusement
			numAmusements[1] += numParcels * 0.2f; // store
			numAmusements[2] += numParcels * 0.2f; // restaurant
		} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {
			numPublics[0] += numParcels * 0.3f;
			numPublics[1] += numParcels * 0.7;
		}
	}
	
	//Block::parcelGraphVertexIter vi, viEnd;
	for (int i = 0; i < blocks.size(); ++i) {
		QVector2D location = QVector2D(blocks.at(i).blockContour.getCentroid());

		// BUG! To be fixed!
		// In some cases, location has very large numbers.
		if (location.x() > 1000000 || location.y() > 1000000) continue;

		// Bounding Boxを取得
		QVector3D minBBox;
		QVector3D maxBBox;
		Polygon3D::getLoopAABB(blocks[i].blockContour.contour, minBBox, maxBBox);

		// 予測される区画数を計算
		int numParcels = blocks[i].blockContour.area() / blocks[i].zone.parcel_area_mean;

		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
			parks.push_back(Office(location, 1, 1));
		} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			// 住人の数を決定
			int num = numParcels * Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.01f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.02f;
			}

			// 人の数を増やす
			int offset = people.size();
			people.resize(people.size() + num);
			for (int pi = offset; pi < people.size(); ++pi) {
				// 家の位置を、ランダムに決定
				people[pi].homeLocation = QVector2D(Util::genRand(minBBox.x(), maxBBox.x()), Util::genRand(minBBox.y(), maxBBox.y()));
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {
			Office office(location, blocks[i].zone.level());
			Office store(location, blocks[i].zone.level());
			Office restaurant(location, blocks[i].zone.level());

			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numCommercials);
				if (type == 0) {
					office.num++;
				} else if (type == 1) {
					store.num++;
				} else {
					restaurant.num++;
				}
				numCommercials[type]--;
			}
			offices.push_back(office);
			stores.push_back(store);
			restaurants.push_back(restaurant);
		} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {
			Office office(location, blocks[i].zone.level());
			Office factory(location, blocks[i].zone.level());

			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numManufacturings);
				if (type == 0) {
					factory.num++;
				} else {
					office.num++;
				}
				numManufacturings[type]--;
			}
			offices.push_back(office);
			factories.push_back(factory);
		} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {
			Office amusement(location, blocks[i].zone.level());
			Office store(location, blocks[i].zone.level());
			Office restaurant(location, blocks[i].zone.level());

			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numAmusements);
				if (type == 0) {
					amusement.num++;
				} else if (type == 1) {
					store.num++;
				} else {
					restaurant.num++;
				}
				numAmusements[type]--;
			}
			amusements.push_back(amusement);
			stores.push_back(store);
			restaurants.push_back(restaurant);
		} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {
			Office school(location, blocks[i].zone.level());
			Office library(location, blocks[i].zone.level());

			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numPublics);
				if (type == 0) {
					library.num++;
				} else {
					school.num++;
				}
				numPublics[type]--;
			}
			schools.push_back(school);
			libraries.push_back(library);
		}
	}

	// 人の好みを割り当てる
	{
		numPeople[0] = people.size() * 0.2f;	// 学生
		numPeople[1] = people.size() * 0.3f;	// 主婦
		numPeople[2] = people.size() * 0.3f;	// サラリーマン
		numPeople[3] = people.size() * 0.2f;	// 老人

		for (int pi = 0; pi < people.size(); ++pi) {
			int type = Util::sampleFromPdf(numPeople);
			numPeople[type]--;

			people[pi].setType(type);
		}
	}

	// put a train station
	{
		stations.push_back(Office(QVector2D(-896, 1232), 1));
	}

	printf("AllocateAll: people=%d, schools=%d, stores=%d, offices=%d, restaurants=%d, amusements=%d, parks=%d, libraries=%d, factories=%d\n", people.size(), schools.size(), stores.size(), offices.size(), restaurants.size(), amusements.size(), parks.size(), libraries.size(), factories.size());
}

/**
 * 住民を配備する
 */
void UrbanGeometry::allocatePeople() {
	people.clear();

	/*
	for (int i = 0; i < blocks.size(); ++i) {
		QVector2D location = QVector2D(blocks.at(i).blockContour.getCentroid());

		// 予測される区画数を計算
		int numParcels = blocks[i].blockContour.area() / blocks[i].zone.parcel_area_mean;

		if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			// 住人の数を決定
			int num = numParcels * Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.01f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.02f;
			}
			numPeople[0] += num * 0.2f;
			numPeople[1] += num * 0.3f;
			numPeople[2] += num * 0.3f;
			numPeople[3] += num * 0.2f;
		}
	}
	*/
	
	//Block::parcelGraphVertexIter vi, viEnd;
	for (int i = 0; i < blocks.size(); ++i) {
		QVector2D location = QVector2D(blocks.at(i).blockContour.getCentroid());
		// BUG! To be fixed!
		// In some cases, location has very large numbers.
		if (location.x() > 1000000 || location.y() > 1000000) continue;

		// Bounding Boxを取得
		QVector3D minBBox;
		QVector3D maxBBox;
		Polygon3D::getLoopAABB(blocks[i].blockContour.contour, minBBox, maxBBox);

		// 予測される区画数を計算
		int numParcels = blocks[i].blockContour.area() / blocks[i].zone.parcel_area_mean;

		if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			// 住人の数を決定
			int num = numParcels * Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.01f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.02f;
			}

			// 人の数を増やす
			int offset = people.size();
			people.resize(people.size() + num);
			for (int pi = offset; pi < people.size(); ++pi) {
				// 家の位置を、ランダムに決定
				people[pi].homeLocation = QVector2D(Util::genRand(minBBox.x(), maxBBox.x()), Util::genRand(minBBox.y(), maxBBox.y()));
			}
		}
	}

	// 人の好みを割り当てる
	{
		std::vector<float> numPeople(4, 0.0f);
		numPeople[0] = people.size() * 0.2f;	// 学生
		numPeople[1] = people.size() * 0.3f;	// 主婦
		numPeople[2] = people.size() * 0.3f;	// サラリーマン
		numPeople[3] = people.size() * 0.2f;	// 老人

		for (int pi = 0; pi < people.size(); ++pi) {
			int type = Util::sampleFromPdf(numPeople);
			numPeople[type]--;

			people[pi].setType(type);
		}
	}
}

/**
 * 各住人にとっての、都市のfeatureベクトルを計算する。結果は、各personのfeatureに格納される。
 * また、それに対する評価結果を、各personのscoreに格納する。
 * さらに、全住人によるscoreの平均を返却する。
 * この関数は、レイヤー情報を使うのでちょっと速い代わりに、直近の店IDなどが分からない。
 * なので、この関数は、findBest()からコールされるべきだ。
 */
float UrbanGeometry::computeScore(VBORenderManager& renderManager) {
	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i], renderManager);
		score_total += people[i].score;
	}

	return score_total / people.size();
}

/**
 * 各住人にとっての、都市のfeatureベクトルを計算する。結果は、各personのfeatureに格納される。
 * また、それに対する評価結果を、各personのscoreに格納する。
 * さらに、全住人によるscoreの平均を返却する。
 * この関数は、直近の店IDなどが分かる代わり、ちょっと遅い。
 * なので、この関数は、loadZoning()からコールされるべきだ。
 */
float UrbanGeometry::computeScore() {
	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i]);
		score_total += people[i].score;
	}

	return score_total / people.size();
}

/**
 * 指定された人の家に最も近い店、学校、レストラン、公園などの距離を計算する。
 * レイヤー情報を使うので、ちょっと速いはず。
 * ただし、直近の店ID、学校ID、レストランIDなどは分からないので、-1としておく。
 * というわけで、この関数は、findBest()からコールされるべきだ。
 * loadZoning()からは、もう一方の関数をコールすべき。
 */
void UrbanGeometry::setFeatureForPerson(Person& person, VBORenderManager& renderManager) {
	person.feature.resize(9);

	person.feature[0] = renderManager.vboStoreLayer.layer.getValue(person.homeLocation);
	person.feature[1] = renderManager.vboSchoolLayer.layer.getValue(person.homeLocation);
	person.feature[2] = renderManager.vboRestaurantLayer.layer.getValue(person.homeLocation);
	person.feature[3] = renderManager.vboParkLayer.layer.getValue(person.homeLocation);
	person.feature[4] = renderManager.vboAmusementLayer.layer.getValue(person.homeLocation);
	person.feature[5] = renderManager.vboLibraryLayer.layer.getValue(person.homeLocation);
	person.feature[6] = renderManager.vboNoiseLayer.layer.getValue(person.homeLocation);
	person.feature[7] = renderManager.vboPollutionLayer.layer.getValue(person.homeLocation);
	person.feature[8] = renderManager.vboStationLayer.layer.getValue(person.homeLocation);
	person.score = std::inner_product(std::begin(person.feature), std::end(person.feature), std::begin(person.preference), 0.0);

	person.nearestStore = -1;
	person.nearestSchool = -1;
	person.nearestRestaurant = -1;
	person.nearestPark = -1;
	person.nearestAmusement = -1;
	person.nearestLibrary = -1;
	person.nearestStation = -1;
}

/**
 * 指定された人の家に最も近い店、学校、レストラン、公園などの距離を計算する。
 * レイヤー情報を使わない代わり、具体的に、直近の店ID、直近の学校ID、直近のレストランIDを計算して保存するので、
 * 人をクリックした際に、直近の店、学校、レストランなどを表示できる。
 * というわけで、この関数は、loadZoning()からコールされるべきだ。
 * findBest()からは、もう一方の関数をコールすべき。そっちの方がちょっと速いから。
 */
void UrbanGeometry::setFeatureForPerson(Person& person) {
	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001};

	person.feature.resize(9);

	{ // nearest store
		std::pair<int, float> n = nearestStore(person.homeLocation);
		person.nearestStore = n.first;
		person.feature[0] = expf(-K[0] * n.second);
	}

	{ // nearest school
		std::pair<int, float> n = nearestSchool(person.homeLocation);
		person.nearestSchool = n.first;
		person.feature[1] = expf(-K[1] * n.second);
	}

	{ // nearest restaurant
		std::pair<int, float> n = nearestRestaurant(person.homeLocation);
		person.nearestRestaurant = n.first;
		person.feature[2] = expf(-K[2] * n.second);
	}

	{ // nearest park
		std::pair<int, float> n = nearestPark(person.homeLocation);
		person.nearestPark = n.first;
		person.feature[3] = expf(-K[3] * n.second);
	}

	{ // nearest amusement
		std::pair<int, float> n = nearestAmusement(person.homeLocation);
		person.nearestAmusement = n.first;
		person.feature[4] = expf(-K[4] * n.second);
	}

	{ // nearest library
		std::pair<int, float> n = nearestLibrary(person.homeLocation);
		person.nearestLibrary = n.first;
		person.feature[5] = expf(-K[5] * n.second);
	}

	{ // noise
		person.feature[6] = expf(-K[6] * noise(person.homeLocation));
	}

	{ // pollution
		person.feature[7] = expf(-K[7] * pollution(person.homeLocation));
	}

	{ // nearest station
		std::pair<int, float> n = nearestStation(person.homeLocation);
		person.nearestStation = n.first;
		person.feature[8] = expf(-K[8] * n.second);
	}

	person.score = std::inner_product(std::begin(person.feature), std::end(person.feature), std::begin(person.preference), 0.0);
}

std::pair<int, float> UrbanGeometry::nearestStore(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestStore = -1;
	for (int i = 0; i < stores.size(); ++i) {
		float dist2 = (stores[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestStore = i;
		}
	}

	return std::make_pair(nearestStore, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestSchool(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestSchool = -1;
	for (int i = 0; i < schools.size(); ++i) {
		float dist2 = (schools[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestSchool = i;
		}
	}

	return std::make_pair(nearestSchool, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestRestaurant(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestRestaurant = -1;
	for (int i = 0; i < restaurants.size(); ++i) {
		float dist2 = (restaurants[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestRestaurant = i;
		}
	}

	return std::make_pair(nearestRestaurant, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestPark(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestPark = -1;
	for (int i = 0; i < parks.size(); ++i) {
		float dist2 = (parks[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestPark = i;
		}
	}

	return std::make_pair(nearestPark, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestAmusement(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestAmusement = -1;
	for (int i = 0; i < amusements.size(); ++i) {
		float dist2 = (amusements[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestAmusement = i;
		}
	}

	return std::make_pair(nearestAmusement, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestLibrary(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestLibrary = -1;
	for (int i = 0; i < libraries.size(); ++i) {
		float dist2 = (libraries[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestLibrary = i;
		}
	}

	return std::make_pair(nearestLibrary, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestFactory(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestFactory = -1;
	for (int i = 0; i < factories.size(); ++i) {
		float dist2 = (factories[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestFactory = i;
		}
	}

	return std::make_pair(nearestFactory, sqrtf(min_dist2));
}

std::pair<int, float> UrbanGeometry::nearestStation(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestStation = -1;
	for (int i = 0; i < stations.size(); ++i) {
		float dist2 = (stations[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestStation = i;
		}
	}

	return std::make_pair(nearestStation, sqrtf(min_dist2));
}

float UrbanGeometry::noise(const QVector2D& pt) {
	float Km = 800.0 - nearestFactory(pt).second;
	float Ka = 400.0 - nearestAmusement(pt).second;
	float Ks = 200.0 - nearestStore(pt).second;

	return std::max(std::max(std::max(Km, Ka), Ks), 0.0f);
}

float UrbanGeometry::pollution(const QVector2D& pt) {
	float Km = 800.0 - nearestFactory(pt).second;

	return std::max(Km, 0.0f);
}

int UrbanGeometry::findNearestPerson(const QVector2D& pt) {
	float min_dist = std::numeric_limits<float>::max();
	int id = -1;

	for (int i = 0; i < people.size(); ++i) {
		float dist = (people[i].homeLocation - pt).lengthSquared();
		if (dist < min_dist) {
			min_dist = dist;
			id = i;
		}
	}

	return id;
}

void UrbanGeometry::updateLayer(int featureId, VBOLayer& layer) {
	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001};

	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			switch (featureId) {
			case 0: // store
				layer.layer.layerData.at<float>(r, c) = expf(-K[0] * nearestStore(QVector2D(x, y)).second);
				break;
			case 1: // school
				layer.layer.layerData.at<float>(r, c) = expf(-K[1] * nearestSchool(QVector2D(x, y)).second);
				break;
			case 2: // restaurant
				layer.layer.layerData.at<float>(r, c) = expf(-K[2] * nearestRestaurant(QVector2D(x, y)).second);
				break;
			case 3: // park
				layer.layer.layerData.at<float>(r, c) = expf(-K[3] * nearestPark(QVector2D(x, y)).second);
				break;
			case 4: // amusement
				layer.layer.layerData.at<float>(r, c) = expf(-K[4] * nearestAmusement(QVector2D(x, y)).second);
				break;
			case 5: // library
				layer.layer.layerData.at<float>(r, c) = expf(-K[5] * nearestLibrary(QVector2D(x, y)).second);
				break;
			case 6: // noise
				layer.layer.layerData.at<float>(r, c) = expf(-K[6] * noise(QVector2D(x, y)));
				break;
			case 7: // pollution
				layer.layer.layerData.at<float>(r, c) = expf(-K[7] * pollution(QVector2D(x, y)));
				break;
			case 8: // station
				layer.layer.layerData.at<float>(r, c) = expf(-K[8] * nearestStation(QVector2D(x, y)).second);
				break;
			}
		}
	}

	layer.layer.updateTexFromData(0, 1);
}
