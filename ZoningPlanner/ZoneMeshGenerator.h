#pragma once

#include "VBORenderManager.h"
#include "BlockSet.h"

class ZoneMeshGenerator {
public:
	ZoneMeshGenerator() {}

	static void generateZoneMesh(VBORenderManager& rendManager, BlockSet& blocks);
};

