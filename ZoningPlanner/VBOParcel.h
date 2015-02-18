/************************************************************************************************
*		VBO Parcel
*		@author igarciad
************************************************************************************************/
#pragma once

#include <QSettings>
#include "VBOBuilding.h"
#include <QTextStream>
#include "Polygon3D.h"
#include "ZoneType.h"

/**
* Parcel.
**/
class Parcel {
public:
	Polygon3D parcelContour;
	Polygon3D parcelBuildableAreaContour;

	ZoneType zone;
	Building myBuilding;

public:
	Parcel();

	/** Set Contour */
	inline void setContour(Polygon3D &inContour)
	{
		this->parcelContour = inContour;
		initializeParcel();
	}

	/**
	* Initialize parcel
	**/
	void initializeParcel()
	{
		if(parcelContour.contour.size()>0){
			boost::geometry::correct(parcelContour.contour);
		}
	}

	/** Compute Parcel Buildable Area */
	float computeBuildableArea(float frontSetback, float rearSetback, float sideSetback, std::vector<int> &frontEdges, std::vector<int> &rearEdges, std::vector<int> &sideEdges, Loop3D &pgonInset);
};

