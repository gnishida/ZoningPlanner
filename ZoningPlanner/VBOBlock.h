/************************************************************************************************
*		VBO Block Class
*		@author igarciad
************************************************************************************************/
#pragma once

#include "VBORenderManager.h"
#include "VBOParcel.h"
#include <QVector3D>
#include "Polygon3D.h"
#include "ZoneType.h"

/**
* Block.
**/
class Block {
public:
	std::vector<Parcel> parcels;
	ZoneType zone;
	Polygon3D blockContour;
	Polygon3D sidewalkContour;
	std::vector<float> sidewalkContourRoadsWidths;
	bool valid;

public:
	Block() : valid(true) {}

	void clear();
	//void buildableAreaMock(void);

	static void findParcelFrontAndBackEdges(Block &inBlock, Parcel &inParcel, std::vector<int> &frontEdges,	std::vector<int> &rearEdges, std::vector<int> &sideEdges);
};

