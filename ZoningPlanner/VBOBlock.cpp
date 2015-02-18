/************************************************************************************************
*		VBO Block Class
*		@author igarciad
************************************************************************************************/

#include "VBOBlock.h"
//#include "../RoadGraph/roadGraphVertex.h"
#include <QVector2D>

void Block::clear(void)
{
	this->blockContour.contour.clear();
	this->sidewalkContour.contour.clear();
	this->sidewalkContourRoadsWidths.clear();
	this->myParcels.clear();
}

void Block::buildableAreaMock(void)
{	
	parcelGraphVertexIter vi, viEnd;
	Loop3D pContourCpy;

	for(boost::tie(vi, viEnd) = boost::vertices(myParcels);
		vi != viEnd; ++vi)
	{
		std::vector<int> frontEdges;
		std::vector<int> backEdges;

		pContourCpy = myParcels[*vi].parcelContour.contour;
		myParcels[*vi].parcelContour.contour.clear();

		Polygon3D::cleanLoop(pContourCpy, myParcels[*vi].parcelContour.contour, 1.0f);
		Polygon3D::reorientFace(myParcels[*vi].parcelContour.contour);

		findParcelFrontAndBackEdges(*this, myParcels[*vi], frontEdges, backEdges, backEdges);

		float area = myParcels[*vi].computeBuildableArea(3.0f, 0.0f, 0.5f,
			frontEdges, backEdges, backEdges, pContourCpy);

	}
}

void Block::findParcelFrontAndBackEdges(Block &inBlock, Parcel &inParcel,
	std::vector<int> &frontEdges,
	std::vector<int> &rearEdges,
	std::vector<int> &sideEdges )
{
	QVector3D midPt;
	typedef boost::geometry::model::d2::point_xy<double> point_2d;
	point_2d bg_midPt;
	boost::geometry::model::polygon<point_2d> bg_pgon;

	float dist;
	float minDist;
	const float tol = 0.01f;
	int next;

	frontEdges.clear();
	rearEdges.clear();
	sideEdges.clear();

	float distPtThis;
	float distPtNext;
	float distPtMid;
	float kDistTol = 0.01f;

	for(int i=0; i<inParcel.parcelContour.contour.size(); ++i)
	{
		next = ((i+1))%inParcel.parcelContour.contour.size();

		midPt = 0.5f*(inParcel.parcelContour.contour.at(i) +
			inParcel.parcelContour.contour.at(next));

		distPtThis = Polygon3D::distanceXYToPoint( inBlock.blockContour.contour, 
			inParcel.parcelContour.contour.at(i) );

		distPtNext = Polygon3D::distanceXYToPoint( inBlock.blockContour.contour, 
			inParcel.parcelContour.contour.at(next) );

		distPtMid = Polygon3D::distanceXYToPoint( inBlock.blockContour.contour, 
			midPt );

		int numPtsThatAreClose =
			(int)(distPtThis < kDistTol) + (int)(distPtMid < kDistTol) + (int)(distPtNext < kDistTol);

		bool midPtIsClose = (distPtThis < kDistTol);

		switch(numPtsThatAreClose){
			//if neither one is close to block boundary, then segment is a rear segment
		case 0:
			rearEdges.push_back(i);
			break;
			//if only one between this and next is close to block boundary, then segment is a side segment
		case 1:					
		case 2:
			sideEdges.push_back(i);
			break;
			//if this or next are very close to block boundary, then segment is a front segment
		case 3:
			frontEdges.push_back(i);
			break;
		}

	}

}

void getRoadSegmentGeoRightAndLeft(std::vector<QVector3D> &roadSegGeo,
	std::vector<QVector3D> &roadSegGeoRight, std::vector<QVector3D> &roadSegGeoLeft, float width)
{
	QVector3D ptR;
	QVector3D ptL;
	QVector2D dirVec;

	for(int i=0; i<roadSegGeo.size(); ++i){
		if( (i!=0) && (i!=roadSegGeo.size()-1) ){
			dirVec = QVector2D(roadSegGeo[i+1] - roadSegGeo[i-1]);
		} else if(i==0){
			dirVec = QVector2D(roadSegGeo[i+1] - roadSegGeo[i]);
		} else {
			dirVec = QVector2D(roadSegGeo[i] - roadSegGeo[i-1]);
		}
		QVector2D dirVecPerp(dirVec.y(), -dirVec.x());
		dirVecPerp.normalize();
		ptR = roadSegGeo[i] + 0.5f*width*dirVecPerp;
		ptL = roadSegGeo[i] - 0.5f*width*dirVecPerp;
		roadSegGeoRight.push_back(ptR);
		roadSegGeoLeft.push_back(ptL);
	}
}
