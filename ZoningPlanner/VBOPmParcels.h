/************************************************************************************************
 *		Procedural City Generation: Parcel
 *		@author igarciad
 ************************************************************************************************/

#pragma once

#include "VBOBlock.h"
#include "Zoning.h"

class VBOPmParcels{
public:
	static bool generateParcels(Zoning& zoning, VBORenderManager& rendManager, BlockSet &blocks);

private:
	static void subdivideBlockIntoParcels(Zoning& zoning, Block &block);
	static bool subdivideParcel(Zoning& zoning, Parcel parcel, std::vector<Parcel> &outParcels);
};
