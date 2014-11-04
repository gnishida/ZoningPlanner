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

public:
	UrbanGeometry(MainWindow* mainWin);
	~UrbanGeometry();

	/** getter for width */
	int getWidth() { return width; }

	/** getter for depth */
	int getDepth() { return depth; }

	void clear();
	void clearGeometry();

	//void render(VBORenderManager &vboRenderManager);
	void adaptToTerrain();

	//void newTerrain(int width, int depth, int cellLength);
	//void loadTerrain(const QString &filename);
	//void saveTerrain(const QString &filename);

	void loadRoads(const QString& filename);
	void saveRoads(const QString& filename);
	void clearRoads();

	void loadBlocks(const QString& filename);
	void saveBlocks(const QString& filename);

	void allocateAll();
	void allocateCommputingPlace();
	float computeScore();
	void setFeatureForPerson(Person& person);
	std::pair<int, float> nearestSchool(const Person& person);
	std::pair<int, float> nearestStore(const Person& person);
	std::pair<int, float> nearestRestaurant(const Person& person);
	std::pair<int, float> nearestPark(const Person& person);
	std::pair<int, float> nearestAmusement(const Person& person);
	std::pair<int, float> nearestLibrary(const Person& person);
	float noise(const QVector2D& pt);
	float airpollution(const QVector2D& pt);

	Person findNearestPerson(const QVector2D& pt);
};
