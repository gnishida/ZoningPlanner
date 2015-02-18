/************************************************************************************************
 *		Procedural City Generation
 *		@author igarciad
 ************************************************************************************************/

#include "VBOPm.h"
#include "Polygon3D.h"
#include "BlockSet.h"
#include "Util.h"
#include "PMBuildingHouse.h"
#include "PMBuildingTower.h"
#include "PMBuildingFactory.h"
#include "PMBuildingRB.h"
#include "PMBuildingSchool.h"

bool VBOPm::generateBuildings(VBORenderManager& rendManager, BlockSet& blocks, Zoning& zones) {
	rendManager.removeStaticGeometry("3d_building");
		
	Block::parcelGraphVertexIter vi, viEnd;
	for (int bN = 0; bN < blocks.size(); bN++) {
		if (blocks[bN].zone.type() == ZoneType::TYPE_PARK) continue;//skip those with parks
		for (boost::tie(vi, viEnd) = boost::vertices(blocks[bN].myParcels); vi != viEnd; ++vi) {
			if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_PARK) continue;
			if (blocks[bN].myParcels[*vi].myBuilding.buildingFootprint.contour.size() < 3) continue;

			float c = rand() % 192;
			blocks[bN].myParcels[*vi].myBuilding.color = QColor(c, c, c);
			if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
				if (blocks[bN].myParcels[*vi].zone.level() == 1) {
					PMBuildingHouse::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
				} else {
					PMBuildingTower::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
				}
			} else if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_COMMERCIAL) {
				blocks[bN].myParcels[*vi].myBuilding.subType = rand() % 2;
				PMBuildingRB::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
			} else if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_MANUFACTURING) {
				blocks[bN].myParcels[*vi].myBuilding.subType = rand() % 3;
				PMBuildingFactory::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
			} else if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_PUBLIC) {
				PMBuildingSchool::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
			} else if (blocks[bN].myParcels[*vi].zone.type() == ZoneType::TYPE_AMUSEMENT) {
				PMBuildingRB::generate(rendManager, "3d_building", blocks[bN].myParcels[*vi].myBuilding);
			}

			//int building_type = 1;//placeTypes.myPlaceTypes[blocks[bN].getMyPlaceTypeIdx()].getInt("building_type");
			//VBOGeoBuilding::generateBuilding(rendManager,blocks[bN].myParcels[*vi].myBuilding, building_type);				
		}
	}
	printf("Building generation is done.\n");

	return true;
}

void VBOPm::generateZoningMesh(VBORenderManager& rendManager, BlockSet& blocks) {
	// 3Dモデルを生成する
	rendManager.removeStaticGeometry("zoning");
	for (int i = 0; i < blocks.size(); ++i) {
		if (!blocks[i].valid) continue;
		//blocks[i].adaptToTerrain(&rendManager);
		
		if (blocks[i].zone.type() == ZoneType::TYPE_UNUSED) continue;

		// Blockの3Dモデルを生成（Block表示モードの時にのみ、表示される）
		{
			std::vector<Vertex> vert;

			QColor color;
			int opacity = 192;
			if (i == blocks.selectedBlockIndex) {
				color = QColor(255, 255, 255, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {	// 住宅街は赤色ベース
				color = QColor(255 - (blocks[i].zone.level() - 1) * 70, 0, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {	// 商業地は青色ベース
				color = QColor(0, 0, 255 - (blocks[i].zone.level() - 1) * 70, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {	// 工場街は灰色ベース
				color = QColor(200 - (blocks[i].zone.level() - 1) * 60, 150 - (blocks[i].zone.level() - 1) * 40, 200 - (blocks[i].zone.level() - 1) * 60, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {			// 公園は緑色
				color = QColor(0, 255, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {		// 繁華街は黄色
				color = QColor(255, 255, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {		// 公共施設は水色ベース
				color = QColor(0, 255, 255, opacity);
			} else {
				color = QColor(128, 128, 128, opacity);
			}

			rendManager.addStaticGeometry2("zoning", blocks[i].blockContour.contour, 8.0f, false, "", GL_QUADS, 1|mode_AdaptTerrain, QVector3D(1, 1, 1), color);
		}
	}
}
