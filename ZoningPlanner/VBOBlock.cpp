/************************************************************************************************
*		VBO Block Class
*		@author igarciad
************************************************************************************************/

#include "VBOBlock.h"
#include <QVector2D>

void Block::clear() {
	this->blockContour.contour.clear();
	this->sidewalkContour.contour.clear();
	this->sidewalkContourRoadsWidths.clear();
	this->parcels.clear();
}

void Block::findParcelFrontAndBackEdges(Block &inBlock, Parcel &inParcel, std::vector<int> &frontEdges, std::vector<int> &rearEdges, std::vector<int> &sideEdges) {
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

	for (int i = 0; i < inParcel.parcelContour.contour.size(); ++i) {
		next = ((i+1))%inParcel.parcelContour.contour.size();

		midPt = 0.5f*(inParcel.parcelContour.contour[i] + inParcel.parcelContour.contour[next]);

		distPtThis = Polygon3D::distanceXYToPoint(inBlock.blockContour.contour, inParcel.parcelContour.contour[i]);

		distPtNext = Polygon3D::distanceXYToPoint(inBlock.blockContour.contour, inParcel.parcelContour.contour[next]);

		distPtMid = Polygon3D::distanceXYToPoint(inBlock.blockContour.contour, midPt);

		int numPtsThatAreClose = (int)(distPtThis < kDistTol) + (int)(distPtMid < kDistTol) + (int)(distPtNext < kDistTol);

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
