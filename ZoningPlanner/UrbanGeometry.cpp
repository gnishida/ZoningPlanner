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
	VBOPm::generateBlockModels(mainWin->glWidget->vboRenderManager, roads, blocks, mainWin->urbanGeometry->zones);
	VBOPm::generateParcelModels(mainWin->glWidget->vboRenderManager, blocks, mainWin->urbanGeometry->zones);
}

void UrbanGeometry::loadBlocks(const QString& filename) {
	blocks.load(filename);
	VBOPmBlocks::assignZonesToBlocks(zones, blocks);
	VBOPm::generateBlockModels(mainWin->glWidget->vboRenderManager, roads, blocks, mainWin->urbanGeometry->zones);
	VBOPm::generateParcelModels(mainWin->glWidget->vboRenderManager, blocks, mainWin->urbanGeometry->zones);
}

void UrbanGeometry::saveBlocks(const QString& filename) {
	blocks.save(filename);
}

/**
 * 住民を配備する
 */
void UrbanGeometry::allocateAll() {
	people.clear();
	offices.clear();
	
	Block::parcelGraphVertexIter vi, viEnd;
	for (int i = 0; i < blocks.size(); ++i) {
		for (boost::tie(vi, viEnd) = boost::vertices(blocks[i].myParcels); vi != viEnd; ++vi) {
			QVector2D location = QVector2D(blocks[i].myParcels[*vi].myBuilding.buildingFootprint.getCentroid());

			if (blocks[i].myParcels[*vi].zone.type == ZoneType::TYPE_RESIDENTIAL) {
				int num = Util::genRand(1, 5);
				num = 1;
				if (blocks[i].myParcels[*vi].zone.level == 2) {
					num = blocks[i].myParcels[*vi].myBuilding.buildingFootprint.area() * 0.05f;
				} else if (blocks[i].myParcels[*vi].zone.level == 3) {
					num = blocks[i].myParcels[*vi].myBuilding.buildingFootprint.area() * 0.20f;
				}

				QVector3D size;
				QMatrix4x4 xformMat;
				Polygon3D::getLoopOBB(blocks[i].myParcels[*vi].myBuilding.buildingFootprint.contour, size, xformMat);

				if (Util::genRand(0, 1) < 0.01) {
					schools.push_back(Office(location));
				} else {					
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
				}
			} else if (blocks[i].myParcels[*vi].zone.type == ZoneType::TYPE_COMMERCIAL) {
				float r = Util::genRand(0, 1);
				if (r < 0.6) {
					offices.push_back(Office(location));
				} else if (r < 0.8) {
					stores.push_back(Office(location));
				} else {
					restaurants.push_back(Office(location));
				}
			} else if (blocks[i].myParcels[*vi].zone.type == ZoneType::TYPE_MANUFACTURING) {
				offices.push_back(Office(location));
			} else if (blocks[i].myParcels[*vi].zone.type == ZoneType::TYPE_AMUSEMENT) {
				float r = Util::genRand(0, 1);
				if (r < 0.6) {
					amusements.push_back(Office(location));
				} else if (r < 0.8) {
					stores.push_back(Office(location));
				} else {
					restaurants.push_back(Office(location));
				}
			} else if (blocks[i].myParcels[*vi].zone.type == ZoneType::TYPE_PARK) {
				parks.push_back(Office(location));
			}
		}
	}

	// if no school, put a school
	if (schools.size() == 0) {
		schools.push_back(Office(QVector2D(0, 0)));
	}

	// allocate commuting office
	for (int i = 0; i < people.size(); ++i) {
		if (people[i].type == Person::TYPE_STUDENT) {
			people[i].commuteTo = Util::genRand(0, schools.size());
		} else if (people[i].type == Person::TYPE_OFFICEWORKER) {
			people[i].commuteTo = Util::genRand(0, offices.size());
		}
	}

	// find the nearest park
	for (int i = 0; i < people.size(); ++i) {
		float min_dist2 = std::numeric_limits<float>::max();
		int nearestPark = -1;
		for (int j = 0; j < parks.size(); ++j) {
			float dist2 = (parks[j].location - people[i].homeLocation).lengthSquared();
			if (dist2 < min_dist2) {
				min_dist2 = dist2;
				nearestPark = j;
			}
		}

		people[i].nearestPark = nearestPark;
	}

	// find the nearest store
	for (int i = 0; i < people.size(); ++i) {
		float min_dist2 = std::numeric_limits<float>::max();
		int nearestStore = -1;
		for (int j = 0; j < stores.size(); ++j) {
			float dist2 = (stores[j].location - people[i].homeLocation).lengthSquared();
			if (dist2 < min_dist2) {
				min_dist2 = dist2;
				nearestStore = j;
			}
		}

		people[i].nearestStore = nearestStore;
	}

	// find the nearest restaurant
	for (int i = 0; i < people.size(); ++i) {
		float min_dist2 = std::numeric_limits<float>::max();
		int nearestRestaurant = -1;
		for (int j = 0; j < restaurants.size(); ++j) {
			float dist2 = (restaurants[j].location - people[i].homeLocation).lengthSquared();
			if (dist2 < min_dist2) {
				min_dist2 = dist2;
				nearestRestaurant = j;
			}
		}

		people[i].nearestRestaurant = nearestRestaurant;
	}

	// find the nearest amusement
	for (int i = 0; i < people.size(); ++i) {
		float min_dist2 = std::numeric_limits<float>::max();
		int nearestAmusement = -1;
		for (int j = 0; j < amusements.size(); ++j) {
			float dist2 = (amusements[j].location - people[i].homeLocation).lengthSquared();
			if (dist2 < min_dist2) {
				min_dist2 = dist2;
				nearestAmusement = j;
			}
		}

		people[i].nearestAmusement = nearestAmusement;
	}

	// compute the avg commuting distance
	{
		float dist_total = 0.0f;
		int count = 0;
		for (int i = 0; i < people.size(); ++i) {
			if (people[i].type == Person::TYPE_STUDENT && people[i].commuteTo >= 0) {
				dist_total += (people[i].homeLocation - schools[people[i].commuteTo].location).length();
				count++;
			} else if (people[i].type == Person::TYPE_OFFICEWORKER && people[i].commuteTo >= 0) {
				dist_total += (people[i].homeLocation - offices[people[i].commuteTo].location).length();
				count++;
			}
		}

		printf("avg commuting distance: %lf\n", dist_total / (float)count);
	}

	// compute the avg distance to the nearest park
	{
		float dist_total = 0.0f;
		int count = 0;
		for (int i = 0; i < people.size(); ++i) {
			if (people[i].nearestPark >= 0) {
				dist_total += (people[i].homeLocation - parks[people[i].nearestPark].location).length();
				count++;
			}
		}

		printf("avg distance to the nearest park: %lf\n", dist_total / (float)count);
	}

	// compute the avg distance to the nearest store
	{
		float dist_total = 0.0f;
		int count = 0;
		for (int i = 0; i < people.size(); ++i) {
			if (people[i].nearestStore >= 0) {
				dist_total += (people[i].homeLocation - stores[people[i].nearestStore].location).length();
				count++;
			}
		}

		printf("avg distance to the nearest store: %lf\n", dist_total / (float)count);
	}

	// compute the avg distance to the nearest restaurant
	{
		float dist_total = 0.0f;
		int count = 0;
		for (int i = 0; i < people.size(); ++i) {
			if (people[i].nearestRestaurant >= 0) {
				dist_total += (people[i].homeLocation - restaurants[people[i].nearestRestaurant].location).length();
				count++;
			}
		}

		printf("avg distance to the nearest restaurant: %lf\n", dist_total / (float)count);
	}
}