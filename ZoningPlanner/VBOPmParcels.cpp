/************************************************************************************************
 *		Procedural City Generation: Parcel
 *		@author igarciad
 ************************************************************************************************/

#include "VBOPmParcels.h"
#include "qmatrix4x4.h"
#include "Util.h"

bool VBOPmParcels::generateParcels(Zoning& zoning, VBORenderManager& rendManager, BlockSet &blocks) {
	srand(0);
	for (int i = 0; i < blocks.size(); ++i) {
		if (blocks[i].valid) {
			subdivideBlockIntoParcels(zoning, blocks[i]);
		}
	}

	return true;
}

void VBOPmParcels::subdivideBlockIntoParcels(Zoning& zoning, Block &block) {
	//Empty parcels in block
	block.parcels.clear();

	//Make the initial parcel of the block be the block itself
	Parcel tmpParcel;
	tmpParcel.setContour(block.blockContour);

	subdivideParcel(zoning, tmpParcel, block.parcels);
}

/**
* Parcel subdivision
* @desc: Recursive subdivision of a parcel using OBB technique
* @return: true if parcel was successfully subdivided, false otherwise
* @areaMean: mean target area of parcels after subdivision
* @areaVar: variance of parcels area after subdivision (normalized in 0-1)
* @splitIrregularity: A normalized value 0-1 indicating how far
*					from the middle point the split line should be
**/
bool VBOPmParcels::subdivideParcel(Zoning& zoning, Parcel parcel, std::vector<Parcel> &outParcels) {
	BBox3D bbox;
	parcel.parcelContour.getBBox3D(bbox.minPt, bbox.maxPt);
	ZoneType z = zoning.getZone(QVector2D(bbox.midPt()));

	float thresholdArea = z.parcel_area_mean + z.parcel_area_deviation * z.parcel_area_mean * Util::genRand(-1, 1);
	
	if (parcel.parcelContour.area() <= std::max(thresholdArea, z.parcel_area_min)) {
		parcel.zone = z;
		outParcels.push_back(parcel);
		return true;
	}

	//compute OBB
	QVector3D obbSize;
	QMatrix4x4 obbMat;
	parcel.parcelContour.getMyOBB(obbSize, obbMat);

	//compute split line passing through center of OBB TODO (+/- irregularity)
	//		and with direction parallel/perpendicular to OBB main axis
	QVector3D slEndPoint;
	QVector3D dirVectorInit, dirVector, dirVectorOrthogonal;
	QVector3D midPt(0.0f, 0.0f, 0.0f);
	QVector3D auxPt(1.0f, 0.0f, 0.0f);
	QVector3D midPtNoise(0.0f, 0.0f, 0.0f);
	std::vector<QVector3D> splitLine;	

	midPt = midPt*obbMat;

	dirVectorInit = (auxPt*obbMat - midPt);
	dirVectorInit.normalize();
	if(obbSize.x() > obbSize.y()){
		dirVector.setX( -dirVectorInit.y() );
		dirVector.setY(  dirVectorInit.x() );
	} else {
		dirVector = dirVectorInit;
	}

	midPtNoise.setX(z.parcel_split_deviation * Util::genRand(-10, 10));
	midPtNoise.setY(z.parcel_split_deviation * Util::genRand(-10, 10));
	midPt = midPt + midPtNoise;

	slEndPoint = midPt + 10000.0f*dirVector;
	splitLine.push_back(slEndPoint);
	slEndPoint = midPt - 10000.0f*dirVector;
	splitLine.push_back(slEndPoint);

	//split parcel with line and obtain two new parcels
	Polygon3D pgon1, pgon2;

	float kDistTol = 0.01f;

	std::vector<Polygon3D> pgons;

	// 簡易版の分割（しょぼいが、速い）
	if (parcel.parcelContour.splitMeWithPolyline(splitLine, pgon1.contour, pgon2.contour)) {
		Parcel parcel1;
		Parcel parcel2;

		parcel1.setContour(pgon1);
		parcel2.setContour(pgon2);

		//call recursive function for both parcels
		subdivideParcel(zoning, parcel1, outParcels);
		subdivideParcel(zoning, parcel2, outParcels);
	} else {
		// CGAL版の分割（遅いが、優れている）
		if (parcel.parcelContour.split(splitLine, pgons)) {
			for (int i = 0; i < pgons.size(); ++i) {
				Parcel parcel;
				parcel.setContour(pgons[i]);

				subdivideParcel(zoning, parcel, outParcels);
			}
		} else {
			parcel.zone = ZoneType(ZoneType::TYPE_PARK, 1);
			outParcels.push_back(parcel);
		}
	}

	return true;
}
