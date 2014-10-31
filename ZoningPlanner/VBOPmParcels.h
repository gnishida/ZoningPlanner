/************************************************************************************************
 *		Procedural City Generation: Parcel
 *		@author igarciad
 ************************************************************************************************/

#pragma once

#include "VBOBlock.h"

class VBOPmParcels{
public:

	static bool generateParcels(VBORenderManager& rendManager, Zoning& zoning, std::vector< Block > &blocks);

	static void assignPlaceTypeToParcels(Zoning& zoning, Block& blocks);
};
