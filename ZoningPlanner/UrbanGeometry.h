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
	std::vector<Office> offices;
	std::vector<Office> schools;
	std::vector<Office> stores;
	std::vector<Office> restaurants;
	std::vector<Office> parks;
	std::vector<Office> amusements;
	std::vector<Office> libraries;
	std::vector<Office> factories;

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

	void findBestPlan(VBORenderManager& renderManager, std::vector<std::vector<float> >& preference);
	QVector2D findBestPlace(VBORenderManager& renderManager, std::vector<float>& preference);
	void questionnaire();
};
