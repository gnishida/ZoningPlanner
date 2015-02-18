/************************************************************************************************
 *		Procedural City Generation: Parcel
 *		@author igarciad
 ************************************************************************************************/

#include "VBOPmParcels.h"
#include "qmatrix4x4.h"
#include "Util.h"

bool VBOPmParcels::generateParcels(VBORenderManager& rendManager, std::vector< Block > &blocks) {
	srand(0);
	for (int i = 0; i < blocks.size(); ++i) {
		if (blocks[i].valid) {
			subdivideBlockIntoParcels(blocks[i]);
		}
	}

	return true;
}

void VBOPmParcels::subdivideBlockIntoParcels(Block &block) {
	//Empty parcels in block
	block.parcels.clear();

	//Make the initial parcel of the block be the block itself
	Parcel tmpParcel;
	tmpParcel.setContour(block.blockContour);

	subdivideParcel(block, tmpParcel, block.zone.parcel_area_mean, block.zone.parcel_area_min, block.zone.parcel_area_deviation, block.zone.parcel_split_deviation, block.parcels);
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
bool VBOPmParcels::subdivideParcel(Block &block, Parcel parcel, float areaMean, float areaMin, float areaStd, float splitIrregularity, std::vector<Parcel> &outParcels) {
	float thresholdArea = areaMean + areaStd*areaMean*(((float)qrand()/RAND_MAX)*2.0f-1.0f);//LC::misctools::genRand(-1.0f, 1.0f)
	
	if (parcel.parcelContour.area() <= std::max(thresholdArea, areaMin)) {
		parcel.zone = block.zone;
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

	midPtNoise.setX(splitIrregularity * Util::genRand(-10, 10));
	midPtNoise.setY(splitIrregularity * Util::genRand(-10, 10));
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
		subdivideParcel(block, parcel1, areaMean, areaMin, areaStd, splitIrregularity, outParcels);
		subdivideParcel(block, parcel2, areaMean, areaMin, areaStd, splitIrregularity, outParcels);
	} else {
		// CGAL版の分割（遅いが、優れている）
		if (parcel.parcelContour.split(splitLine, pgons)) {
			for (int i = 0; i < pgons.size(); ++i) {
				Parcel parcel;
				parcel.setContour(pgons[i]);

				subdivideParcel(block, parcel, areaMean, areaMin, areaStd, splitIrregularity, outParcels);
			}
		} else {
			parcel.zone = ZoneType(ZoneType::TYPE_PARK, 1);
			outParcels.push_back(parcel);
		}
	}

	return true;
}
