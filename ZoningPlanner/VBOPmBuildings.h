/************************************************************************************************
 *		Procedural City Generation: Buildings
 *		@author igarciad
 ************************************************************************************************/

#pragma once

#include "VBOBlock.h"
#include "Zoning.h"

class VBOPmBuildings {
public:
	static bool generateBuildings(VBORenderManager& rendManager, BlockSet &blocks);
	static bool computeBuildingFootprintPolygon(float maxFrontage, float maxDepth,	std::vector<int> &frontEdges, std::vector<int> &rearEdges, std::vector<int> &sideEdges, Loop3D &buildableAreaCont, Loop3D &buildingFootprint);
	static bool generateParcelBuildings(VBORenderManager& rendManager, Block &inBlock, Parcel &inParcel);
};

