﻿#include "UrbanGeometry.h"
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
	
	//Block::parcelGraphVertexIter vi, viEnd;
	time_t start, end;
	start = clock();
	for (int i = 0; i < blocks.size(); ++i) {
		QVector2D location = QVector2D(blocks.at(i).blockContour.getCentroid());

		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
			parks.push_back(Office(location));
			continue;
		} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			// 住人の数を決定
			int num = Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.05f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.20f;
			}

			for (int n = 0; n < num; ++n) {
				//QVector2D noise(Util::genRand(-size.x() * 0.5, size.x() * 0.5), Util::genRand(-size.y() * 0.5, size.y() * 0.5));
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

				people.push_back(Person(type, location));
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {
			float r = Util::genRand(0, 1);
			if (r < 0.6) {
				offices.push_back(Office(location));
			} else if (r < 0.8) {
				stores.push_back(Office(location));
			} else {
				restaurants.push_back(Office(location));
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {
			factories.push_back(Office(location));
			offices.push_back(Office(location));
		} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {
			float r = Util::genRand(0, 1);
			if (r < 0.6) {
				amusements.push_back(Office(location));
			} else if (r < 0.8) {
				stores.push_back(Office(location));
			} else {
				restaurants.push_back(Office(location));
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {
			float r = Util::genRand(0, 1);
			if (r < 0.3) {
				libraries.push_back(Office(location));
			} else {
				schools.push_back(Office(location));
			}
		}


		/*
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
				// MODIFIED: Since this is a small park, we do not count this as a park.
				//parks.push_back(Office(location));
			} else if (blocks[i].myParcels[*vi].zone.type() == ZoneType::TYPE_PUBLIC) {
				float r = Util::genRand(0, 1);
				if (r < 0.3) {
					libraries.push_back(Office(location));
				} else {
					schools.push_back(Office(location));
				}
			}
		}
		*/
	}

	end = clock();
	printf("%lf\n", (double)(end - start) / CLOCKS_PER_SEC);

	// put a train station
	{
		stations.push_back(Office(QVector2D(-896, 1232)));
	}

	printf("AllocateAll: people=%d, schools=%d, stores=%d, offices=%d, restaurants=%d, amusements=%d, parks=%d, libraries=%d, factories=%d\n", people.size(), schools.size(), stores.size(), offices.size(), restaurants.size(), amusements.size(), parks.size(), libraries.size(), factories.size());

	if (schools.size() == 0 || restaurants.size() == 0 || amusements.size() == 0 || parks.size() == 0 || libraries.size() == 0 || factories.size() == 0) {
		return false;
	} else {
		return true;
	}
}

/**
 * 各住人にとっての、都市のfeatureベクトルを計算する。結果は、各personのfeatureに格納される。
 * また、それに対する評価結果を、各personのscoreに格納する。
 * さらに、全住人によるscoreの平均を返却する。
 */
float UrbanGeometry::computeScore(VBORenderManager& renderManager) {
	QFile file("features.txt");
	file.open(QIODevice::WriteOnly);
	QTextStream out(&file);

	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i], renderManager);
		std::vector<float> f = people[i].feature;

		float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.01, 0.01, 0.001};

		for (int j = 0; j < f.size(); ++j) {
			f[j] = exp(-K[j] * f[j]);
			out << f[j];
			if (j < f.size() - 1) {
				out << ",";
			}
		}
		out << "\n";

		people[i].score = std::inner_product(std::begin(f), std::end(f), std::begin(people[i].preference), 0.0);
		score_total += people[i].score;
	}

	out.flush();
	file.close();

	return score_total / (float)people.size();
}

void UrbanGeometry::setFeatureForPerson(Person& person, VBORenderManager& renderManager) {
	person.feature.clear();
	person.feature.resize(9);

	// 各propertyのsensitivity
	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.01, 0.01, 0.001};

	/*
	person.feature[0] = expf(-K[0] * renderManager.vboStoreLayer.layer.getValue(person.homeLocation));
	person.feature[1] = expf(-K[1] * renderManager.vboSchoolLayer.layer.getValue(person.homeLocation));
	person.feature[2] = expf(-K[2] * renderManager.vboRestaurantLayer.layer.getValue(person.homeLocation));
	person.feature[3] = expf(-K[3] * renderManager.vboParkLayer.layer.getValue(person.homeLocation));
	person.feature[4] = expf(-K[4] * renderManager.vboAmusementLayer.layer.getValue(person.homeLocation));
	person.feature[5] = expf(-K[5] * renderManager.vboLibraryLayer.layer.getValue(person.homeLocation));
	person.feature[6] = expf(-K[6] * renderManager.vboNoiseLayer.layer.getValue(person.homeLocation));
	person.feature[7] = expf(-K[7] * renderManager.vboPollutionLayer.layer.getValue(person.homeLocation));
	person.feature[8] = expf(-K[8] * renderManager.vboStationLayer.layer.getValue(person.homeLocation));
	*/

	{ // nearest store
		std::pair<int, float> n = nearestStore(person.homeLocation);
		person.nearestStore = n.first;
		person.feature[0] = n.second;
	}

	{ // nearest school
		std::pair<int, float> n = nearestSchool(person.homeLocation);
		person.nearestSchool = n.first;
		person.feature[1] = n.second;
	}

	{ // nearest restaurant
		std::pair<int, float> n = nearestRestaurant(person.homeLocation);
		person.nearestRestaurant = n.first;
		person.feature[2] = n.second;
	}

	{ // nearest park
		std::pair<int, float> n = nearestPark(person.homeLocation);
		person.nearestPark = n.first;
		person.feature[3] = n.second;
	}

	{ // nearest amusement
		std::pair<int, float> n = nearestAmusement(person.homeLocation);
		person.nearestAmusement = n.first;
		person.feature[4] = n.second;
	}

	{ // nearest library
		std::pair<int, float> n = nearestLibrary(person.homeLocation);
		person.nearestLibrary = n.first;
		person.feature[5] = n.second;
	}

	{ // noise
		person.feature[6] = noise(person.homeLocation);
	}

	{ // pollution
		person.feature[7] = pollution(person.homeLocation);
	}

	{ // nearest station
		std::pair<int, float> n = nearestStation(person.homeLocation);
		person.nearestStation = n.first;
		person.feature[8] = n.second;
	}
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
