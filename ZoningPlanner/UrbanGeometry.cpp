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

UrbanGeometry::UrbanGeometry(MainWindow* mainWin) {
	this->mainWin = mainWin;

	zones.load("zoning.xml");
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

/*void UrbanGeometry::newTerrain(int width, int depth, int cellLength) {
	clear();
}*/

/*void UrbanGeometry::loadTerrain(const QString &filename) {
	printf("NOT IMPLEMENTED YET\n");
}

void UrbanGeometry::saveTerrain(const QString &filename) {
	printf("NOT IMPLEMENTED YET\n");
}*/

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
 * 住民、オフィス、レストラン、図書館、公園、工場、などなどを配備する
 * １つも配備されない施設があった場合は、falseを返却する。
 */
bool UrbanGeometry::allocateAll() {
	people.clear();
	offices.clear();
	schools.clear();
	stores.clear();
	restaurants.clear();
	amusements.clear();
	parks.clear();
	libraries.clear();
	factories.clear();
	
	Block::parcelGraphVertexIter vi, viEnd;
	for (int i = 0; i < blocks.size(); ++i) {
		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
			QVector2D location = QVector2D(blocks[i].blockContour.getCentroid());
			parks.push_back(Office(location));
			continue;
		}

		for (boost::tie(vi, viEnd) = boost::vertices(blocks[i].myParcels); vi != viEnd; ++vi) {
			QVector2D location = QVector2D(blocks[i].myParcels[*vi].myBuilding.buildingFootprint.getCentroid());

			if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
				int num = Util::genRand(1, 5);
				if (blocks[i].myParcels[*vi].zone.level() == 2) {
					num = blocks[i].myParcels[*vi].myBuilding.buildingFootprint.area() * 0.05f;
				} else if (blocks[i].myParcels[*vi].zone.level() == 3) {
					num = blocks[i].myParcels[*vi].myBuilding.buildingFootprint.area() * 0.20f;
				}

				QVector3D size;
				QMatrix4x4 xformMat;
				Polygon3D::getLoopOBB(blocks[i].myParcels[*vi].myBuilding.buildingFootprint.contour, size, xformMat);

				for (int n = 0; n < num; ++n) {
					QVector2D noise(Util::genRand(-size.x() * 0.5, size.x() * 0.5), Util::genRand(-size.y() * 0.5, size.y() * 0.5));
					float r = Util::genRand(0, 1);
					int type = Person::TYPE_UNKNOWN;
					if (r < 0.2) {
						type = Person::TYPE_STUDENT;
					} else if (r < 0.5) {
						type = Person::TYPE_HOUSEWIFE;
					} else if (r < 0.8) {
						type = Person::TYPE_OFFICEWORKER;
					} else {
						type = Person::TYPE_ELDERLY;
					}

					people.push_back(Person(type, location + noise));
				}
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_COMMERCIAL) {
				float r = Util::genRand(0, 1);
				if (r < 0.6) {
					offices.push_back(Office(location));
				} else if (r < 0.8) {
					stores.push_back(Office(location));
				} else {
					restaurants.push_back(Office(location));
				}
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_MANUFACTURING) {
				factories.push_back(Office(location));
				offices.push_back(Office(location));
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_AMUSEMENT) {
				float r = Util::genRand(0, 1);
				if (r < 0.6) {
					amusements.push_back(Office(location));
				} else if (r < 0.8) {
					stores.push_back(Office(location));
				} else {
					restaurants.push_back(Office(location));
				}
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_PARK) {
				parks.push_back(Office(location));
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_PUBLIC) {
				float r = Util::genRand(0, 1);
				if (r < 0.3) {
					libraries.push_back(Office(location));
				} else {
					schools.push_back(Office(location));
				}
			}
		}
	}

	// put schools outside this region
	{
		schools.push_back(Office(QVector2D(-10000, -10000)));
		schools.push_back(Office(QVector2D(10000, -10000)));
		schools.push_back(Office(QVector2D(10000, 10000)));
		schools.push_back(Office(QVector2D(-10000, 10000)));
	}

	// put libraries outside this region
	{
		libraries.push_back(Office(QVector2D(-10000, -10000)));
		libraries.push_back(Office(QVector2D(10000, -10000)));
		libraries.push_back(Office(QVector2D(10000, 10000)));
		libraries.push_back(Office(QVector2D(-10000, 10000)));
	}

	// put amusements outside this region
	{
		amusements.push_back(Office(QVector2D(-10000, -10000)));
		amusements.push_back(Office(QVector2D(10000, -10000)));
		amusements.push_back(Office(QVector2D(10000, 10000)));
		amusements.push_back(Office(QVector2D(-10000, 10000)));
	}

	// put parks outside this region
	{
		parks.push_back(Office(QVector2D(-10000, -10000)));
		parks.push_back(Office(QVector2D(10000, -10000)));
		parks.push_back(Office(QVector2D(10000, 10000)));
		parks.push_back(Office(QVector2D(-10000, 10000)));
	}

	// put offices outside this region
	{
		offices.push_back(Office(QVector2D(-10000, -10000)));
		offices.push_back(Office(QVector2D(10000, -10000)));
		offices.push_back(Office(QVector2D(10000, 10000)));
		offices.push_back(Office(QVector2D(-10000, 10000)));
	}

	// put a train station
	{
		stations.push_back(Office(QVector2D(-896, 1232)));
	}

	//allocateCommputingPlace();

	printf("AllocateAll: people=%d, schools=%d, stores=%d, offices=%d, restaurants=%d, amusements=%d, parks=%d, libraries=%d, factories=%d\n", people.size(), schools.size(), stores.size(), offices.size(), restaurants.size(), amusements.size(), parks.size(), libraries.size(), factories.size());

	if (schools.size() == 0 || restaurants.size() == 0 || amusements.size() == 0 || parks.size() == 0 || libraries.size() == 0 || factories.size() == 0) {
		return false;
	} else {
		return true;
	}
}

/**
 * Allocate a commuting place to each person.
 */
void UrbanGeometry::allocateCommputingPlace() {
	for (int i = 0; i < people.size(); ++i) {
		if (people[i].type() == Person::TYPE_STUDENT) {
			people[i].commuteTo = Util::genRand(0, schools.size());
		} else if (people[i].type() == Person::TYPE_OFFICEWORKER) {
			people[i].commuteTo = Util::genRand(0, offices.size());
		}
	}
}

/**
 * 各住人にとっての、都市のfeatureベクトルを計算する。結果は、各personのfeatureに格納される。
 * また、それに対する評価結果を、各personのscoreに格納する。
 * さらに、全住人によるscoreの平均を返却する。
 */
float UrbanGeometry::computeScore(VBORenderManager& renderManager) {
	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i], renderManager);
		std::vector<float> f = people[i].feature;

		for (int j = 0; j < f.size(); ++j) {
			f[j] = exp(-f[j] * 0.001);
		}

		people[i].score = std::inner_product(std::begin(f), std::end(f), std::begin(people[i].preference), 0.0);
		score_total += people[i].score;
	}

	return score_total / (float)people.size();
}

void UrbanGeometry::setFeatureForPerson(Person& person, VBORenderManager& renderManager) {
	person.feature.clear();
	person.feature.resize(8);

	person.feature[0] = renderManager.vboStoreLayer.layer.getValue(person.homeLocation);
	person.feature[1] = renderManager.vboSchoolLayer.layer.getValue(person.homeLocation);
	person.feature[2] = renderManager.vboRestaurantLayer.layer.getValue(person.homeLocation);
	person.feature[3] = renderManager.vboParkLayer.layer.getValue(person.homeLocation);
	person.feature[4] = renderManager.vboAmusementLayer.layer.getValue(person.homeLocation);
	person.feature[5] = renderManager.vboLibraryLayer.layer.getValue(person.homeLocation);
	person.feature[6] = renderManager.vboNoiseLayer.layer.getValue(person.homeLocation);
	person.feature[7] = renderManager.vboPollutionLayer.layer.getValue(person.homeLocation);
	person.feature[8] = renderManager.vboStationLayer.layer.getValue(person.homeLocation);



	/*
	person.feature[0] = nearestStore(person.homeLocation).second;
	person.feature[1] = nearestSchool(person.homeLocation).second;
	person.feature[2] = nearestRestaurant(person.homeLocation).second;
	person.feature[3] = nearestPark(person.homeLocation).second;
	person.feature[4] = nearestAmusement(person.homeLocation).second;
	person.feature[5] = nearestLibrary(person.homeLocation).second;
	person.feature[6] = noise(person.homeLocation);
	person.feature[7] = pollution(person.homeLocation);
	person.feature[8] = nearestStation(person.homeLocation).second;
	*/

	person.nearestLibrary = nearestLibrary(person.homeLocation).first;
}

std::pair<int, float> UrbanGeometry::nearestPerson(const QVector2D& pt) {
	float min_dist2 = std::numeric_limits<float>::max();
	int nearestPerson = -1;
	for (int i = 0; i < people.size(); ++i) {
		float dist2 = (people[i].homeLocation - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestPerson = i;
		}
	}

	return std::make_pair(nearestPerson, sqrtf(min_dist2));
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
	int nearestPark = -1;
	for (int i = 0; i < amusements.size(); ++i) {
		float dist2 = (amusements[i].location - pt).lengthSquared();
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			nearestPark = i;
		}
	}

	return std::make_pair(nearestPark, sqrtf(min_dist2));
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

float UrbanGeometry::noise(const QVector2D& pt) {
	float n = 0.0f;
	float Km = 120.0f;
	float Ka = 100.0f;
	float Kc = 100.0f;

	// noise by the manufactoring
	for (int i = 0; i < factories.size(); ++i) {
		float len = (factories[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Km > attenuation) {
				n += Km - attenuation;
			}
		} else {
			n += Km;
		}
	}

	// noise by the amusement facilities
	for (int i = 0; i < amusements.size(); ++i) {
		float len = (amusements[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Ka > attenuation) {
				n += Ka - attenuation;
			}
		} else {
			n += Ka;
		}
	}

	// noise by the commercial stores
	for (int i = 0; i < stores.size(); ++i) {
		float len = (stores[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Kc > attenuation) {
				n += Kc - attenuation;
			}
		} else {
			n += Kc;
		}
	}

	return n;
}

float UrbanGeometry::pollution(const QVector2D& pt) {
	float n = 0.0f;
	float Km = 120.0f;

	// pollution by the manufactoring
	for (int i = 0; i < factories.size(); ++i) {
		float len = (factories[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Km > attenuation) {
				n += Km - attenuation;
			}
		} else {
			n += Km;
		}
	}

	return n;
}

Person UrbanGeometry::findNearestPerson(const QVector2D& pt) {
	float min_dist = std::numeric_limits<float>::max();
	int id = -1;

	for (int i = 0; i < people.size(); ++i) {
		float dist = (people[i].homeLocation - pt).lengthSquared();
		if (dist < min_dist) {
			min_dist = dist;
			id = i;
		}
	}

	return people[id];
}

void UrbanGeometry::updateStationMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestStation(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateStoreMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestStore(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateSchoolMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestSchool(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateRestaurantMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestRestaurant(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateParkMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestPark(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateAmusementMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestAmusement(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateLibraryMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = nearestLibrary(QVector2D(x, y)).second;
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updateNoiseMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = noise(QVector2D(x, y));
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}

void UrbanGeometry::updatePollutionMap(VBOLayer& layer) {
	for (int r = 0; r < layer.layer.layerData.rows; ++r) {
		float y = layer.layer.minPos.y() + (layer.layer.maxPos.y() - layer.layer.minPos.y()) / layer.layer.layerData.rows * r;
		for (int c = 0; c < layer.layer.layerData.cols; ++c) {
			float x = layer.layer.minPos.x() + (layer.layer.maxPos.x() - layer.layer.minPos.x()) / layer.layer.layerData.cols * c;

			layer.layer.layerData.at<float>(r, c) = pollution(QVector2D(x, y));
		}
	}

	layer.layer.updateTexFromData(0, 1000);
}
