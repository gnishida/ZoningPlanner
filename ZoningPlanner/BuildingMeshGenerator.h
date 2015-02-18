#pragma once

#include "VBORenderManager.h"
#include "BlockSet.h"
#include "Zoning.h"

class BuildingMeshGenerator {
public:
	BuildingMeshGenerator() {}

	static bool generateBuildingMesh(VBORenderManager& rendManager, BlockSet& blocks, Zoning& zones);
};

