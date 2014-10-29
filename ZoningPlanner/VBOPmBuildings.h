/************************************************************************************************
 *		Procedural City Generation: Buildings
 *		@author igarciad
 ************************************************************************************************/

#pragma once

#include "VBOPlaceType.h"
#include "VBOBlock.h"
#include "Zoning.h"

class VBOPmBuildings{
public:

	static bool generateBuildings(VBORenderManager& rendManager, Zoning& zoning, std::vector< Block > &blocks);

};

