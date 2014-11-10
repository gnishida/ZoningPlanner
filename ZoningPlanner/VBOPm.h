/************************************************************************************************
 *		Procedural City Generation
 *		@author igarciad
 ************************************************************************************************/

#pragma once

#include <boost/graph/planar_face_traversal.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>

#include "VBOBlock.h"
#include "VBOParcel.h"
#include "VBOBuilding.h"
#include "RoadGraph.h"
#include "BlockSet.h"
#include "Person.h"
#include "UrbanGeometry.h"

class VBORenderManager;

class VBOPm {
public:
	static bool initializedLC;
	static void initLC();

	static bool generateBlocks(VBORenderManager& rendManager, RoadGraph &roadGraph, BlockSet& blocks, Zoning& zones);
	static void generateBlockModels(VBORenderManager& rendManager,RoadGraph &roadGraph, BlockSet& blocks);
	static bool generateParcels(VBORenderManager& rendManager, BlockSet& blocks);
	static void generateParcelModels(VBORenderManager& rendManager, BlockSet& blocks);
	static bool generateBuildings(VBORenderManager& rendManager, BlockSet& blocks, Zoning& zones);
	static bool generateVegetation(VBORenderManager& rendManager, BlockSet& blocks, Zoning& zones);

	static void generateZoningMesh(VBORenderManager& rendManager, BlockSet& blocks);
	static void generatePeopleMesh(VBORenderManager& rendManager, std::vector<Person>& people);
	static void generateSelecionMesh(VBORenderManager& rendManager, Person& person, UrbanGeometry& urbanGeometry);
};


