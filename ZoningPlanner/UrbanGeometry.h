#pragma once

#include "glew.h"

#include "VBORenderManager.h"
#include "RoadGraph.h"
#include "BlockSet.h"
#include "Zoning.h"
#include "Person.h"
#include "Office.h"

class MainWindow;

class UrbanGeometry {
public:
	int width;
	int depth;
	MainWindow* mainWin;

	Zoning zones;
	RoadGraph roads;
	BlockSet blocks;
	std::vector<Person> people;
	std::vector<Office> offices;
	std::vector<Office> schools;
	std::vector<Office> stores;
	std::vector<Office> restaurants;
	std::vector<Office> parks;
	std::vector<Office> amusements;
	std::vector<Office> libraries;
	std::vector<Office> factories;
	std::vector<Office> stations;
	Layer noiseMap;

	int selectedPerson;
	int selectedStore;
	int selectedSchool;
	int selectedRestaurant;
	int selectedPark;
	int selectedAmusement;
	int selectedLibrary;

public:
	UrbanGeometry(MainWindow* mainWin);
	~UrbanGeometry();

	/** getter for width */
	int getWidth() { return width; }

	/** getter for depth */
	int getDepth() { return depth; }

	void clear();
	void clearGeometry();

	void adaptToTerrain();

	void loadRoads(const QString& filename);
	void saveRoads(const QString& filename);
	void clearRoads();

	void loadBlocks(const QString& filename);
	void saveBlocks(const QString& filename);

	void findBestPlan(VBORenderManager& renderManager, int numIterations);
	void findBestPlanGPU(VBORenderManager& renderManager, int numIterations);

	void allocateAll();
	void allocatePeople();
	float computeScore(VBORenderManager& renderManager);
	float computeScore();
	void setFeatureForPerson(Person& person, VBORenderManager& renderManager);
	void setFeatureForPerson(Person& person);

	void movePeople(VBORenderManager& renderManager);
	void movePeopleMT(VBORenderManager& renderManager);
	void movePeopleGPU(VBORenderManager& renderManager);

	std::pair<int, float> nearestPerson(const QVector2D& pt);
	std::pair<int, float> nearestStore(const QVector2D& pt);
	std::pair<int, float> nearestSchool(const QVector2D& pt);
	std::pair<int, float> nearestRestaurant(const QVector2D& pt);
	std::pair<int, float> nearestPark(const QVector2D& pt);
	std::pair<int, float> nearestAmusement(const QVector2D& pt);
	std::pair<int, float> nearestLibrary(const QVector2D& pt);
	std::pair<int, float> nearestFactory(const QVector2D& pt);
	std::pair<int, float> nearestStation(const QVector2D& pt);
	float noise(const QVector2D& pt);
	float pollution(const QVector2D& pt);

	int findNearestPerson(const QVector2D& pt);

	void updateLayer(int featureId, VBOLayer& layer);
};
