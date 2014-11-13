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

		// 予測される区画数を計算
		int numParcels = blocks[i].blockContour.area() / blocks[i].zone.parcel_area_mean;

		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
			parks.push_back(Office(location, 1));
		} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
			// 住人の数を決定
			int num = numParcels * Util::genRand(1, 5);
			if (blocks[i].zone.level() == 2) {
				num = blocks[i].blockContour.area() * 0.01f;
			} else if (blocks[i].zone.level() == 3) {
				num = blocks[i].blockContour.area() * 0.02f;
			}

			for (int n = 0; n < num; ++n) {
				int type = Util::sampleFromPdf(numPeople);
				numPeople[type]--;

				people.push_back(Person(type, location));
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {
			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numCommercials);
				if (type == 0) {
					offices.push_back(Office(location, blocks[i].zone.level()));
				} else if (type == 1) {
					stores.push_back(Office(location, blocks[i].zone.level()));
				} else {
					restaurants.push_back(Office(location, blocks[i].zone.level()));
				}
				numCommercials[type]--;
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {
			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numManufacturings);
				if (type == 0) {
					factories.push_back(Office(location, blocks[i].zone.level()));
				} else {
					offices.push_back(Office(location, blocks[i].zone.level()));
				}
				numManufacturings[type]--;
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {
			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numAmusements);
				if (type == 0) {
					amusements.push_back(Office(location, blocks[i].zone.level()));
				} else if (type == 1) {
					stores.push_back(Office(location, blocks[i].zone.level()));
				} else {
					restaurants.push_back(Office(location, blocks[i].zone.level()));
				}
				numAmusements[type]--;
			}
		} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {
			for (int n = 0; n < numParcels; ++n) {
				int type = Util::sampleFromPdf(numPublics);
				if (type == 0) {
					libraries.push_back(Office(location, blocks[i].zone.level()));
				} else {
					schools.push_back(Office(location, blocks[i].zone.level()));
				}
				numPublics[type]--;
			}
		}
	}

	// put a train station
	{
		stations.push_back(Office(QVector2D(-896, 1232), 1));
	}

	//printf("AllocateAll: people=%d, schools=%d, stores=%d, offices=%d, restaurants=%d, amusements=%d, parks=%d, libraries=%d, factories=%d\n", people.size(), schools.size(), stores.size(), offices.size(), restaurants.size(), amusements.size(), parks.size(), libraries.size(), factories.size());
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
		std::vector<float> f = people[i].feature;

		float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.01, 0.01, 0.001};

		for (int j = 0; j < f.size(); ++j) {
			f[j] = exp(-K[j] * f[j]);
		}

		people[i].score = std::inner_product(std::begin(f), std::end(f), std::begin(people[i].preference), 0.0);
		score_total += people[i].score;
	}

	return score_total / (float)people.size();
}

/**
 * 各住人にとっての、都市のfeatureベクトルを計算する。結果は、各personのfeatureに格納される。
 * また、それに対する評価結果を、各personのscoreに格納する。
 * さらに、全住人によるscoreの平均を返却する。
 * この関数は、直近の店IDなどが分かる代わり、ちょっと遅い。
 * なので、この関数は、loadZoning()からコールされるべきだ。
 */
float UrbanGeometry::computeScore() {
	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.01, 0.01, 0.001};

	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i]);
		std::vector<float> f = people[i].feature;

		for (int j = 0; j < f.size(); ++j) {
			f[j] = exp(-K[j] * f[j]);
		}

		people[i].score = std::inner_product(std::begin(f), std::end(f), std::begin(people[i].preference), 0.0);
		score_total += people[i].score;
	}

	return score_total / (float)people.size();
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
	person.feature.resize(9);

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

/**
 * 人を動かす
 */
void UrbanGeometry::movePeople(VBORenderManager& renderManager) {
	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.01, 0.01, 0.001};

	float score_total = 0.0f;
	for (int i = 0; i < people.size(); ++i) {
		setFeatureForPerson(people[i], renderManager);
	}


	for (int loop = 0; loop < 10; ++loop) {
		for (int i = 0; i < people.size(); ++i) {
			std::vector<float> f1 = people[i].feature;

			for (int j = 0; j < f1.size(); ++j) {
				f1[j] = exp(-K[j] * f1[j]);
			}

			float max_increase = 0.0f;
			int swap_id = -1;
			for (int j = i + 1; j < people.size(); ++j) {
				if (people[i].type() == people[j].type()) continue;

				std::vector<float> f2 = people[j].feature;
				for (int k = 0; k < f2.size(); ++k) {
					f2[k] = exp(-K[k] * f2[k]);
				}

				float increase = std::inner_product(std::begin(f1), std::end(f1), std::begin(people[j].preference), 0.0)
					+ std::inner_product(std::begin(f2), std::end(f2), std::begin(people[i].preference), 0.0)
					- std::inner_product(std::begin(f1), std::end(f1), std::begin(people[i].preference), 0.0)
					- std::inner_product(std::begin(f2), std::end(f2), std::begin(people[j].preference), 0.0);
				if (increase > max_increase) {
					max_increase = increase;
					swap_id = j;
				}
			}

			if (swap_id >= 0) {
				/*
				std::vector<float> f2 = people[swap_id].feature;
				for (int j = 0; j < f2.size(); ++j) {
					f2[j] = exp(-K[j] * f2[j]);
				}
				std::vector<float> p2 = people[swap_id].preference;

				float x1 = std::inner_product(std::begin(f1), std::end(f1), std::begin(people[i].preference), 0.0);
				float x2 = std::inner_product(std::begin(f2), std::end(f2), std::begin(people[swap_id].preference), 0.0);
				float x3 = std::inner_product(std::begin(f1), std::end(f1), std::begin(people[swap_id].preference), 0.0);
				float x4 = std::inner_product(std::begin(f2), std::end(f2), std::begin(people[i].preference), 0.0);
				*/

				std::swap(people[i].homeLocation, people[swap_id].homeLocation);
				std::swap(people[i].feature, people[swap_id].feature);
			}
		}
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
		if (factories[i].level == 0) {
			printf("ERROR!!!!! level = 0!!!\n");
		}

		float len = (factories[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Km > attenuation) {
				n += factories[i].level * (Km - attenuation);
			}
		} else {
			n += factories[i].level * Km;
		}
	}

	// noise by the amusement facilities
	for (int i = 0; i < amusements.size(); ++i) {
		if (amusements[i].level == 0) {
			printf("ERROR!!!!! level = 0!!!\n");
		}

		float len = (amusements[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Ka > attenuation) {
				n += amusements[i].level * (Ka - attenuation);
			}
		} else {
			n += amusements[i].level * Ka;
		}
	}

	// noise by the commercial stores
	for (int i = 0; i < stores.size(); ++i) {
		if (stores[i].level == 0) {
			printf("ERROR!!!!! level = 0!!!\n");
		}

		float len = (stores[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Kc > attenuation) {
				n += stores[i].level * (Kc - attenuation);
			}
		} else {
			n += stores[i].level * Kc;
		}
	}

	return n;
}

float UrbanGeometry::pollution(const QVector2D& pt) {
	float n = 0.0f;
	float Km = 120.0f;

	// pollution by the manufactoring
	for (int i = 0; i < factories.size(); ++i) {
		if (factories[i].level == 0) {
			printf("ERROR!!!!! level = 0!!!\n");
		}

		float len = (factories[i].location - pt).length();
		if (len > 1.0f) {
			float attenuation = 20 * logf(len);
			if (Km > attenuation) {
				n += factories[i].level * (Km - attenuation);
			}
		} else {
			n += factories[i].level * Km;
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
