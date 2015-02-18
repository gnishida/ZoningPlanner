#include "BuildingMeshGenerator.h"
#include "PMBuildingFactory.h"
#include "PMBuildingHouse.h"
#include "PMBuildingRB.h"
#include "PMBuildingSchool.h"
#include "PMBuildingTower.h"

bool BuildingMeshGenerator::generateBuildingMesh(VBORenderManager& rendManager, BlockSet& blocks, Zoning& zones) {
	rendManager.removeStaticGeometry("3d_building");
		
	for (int bN = 0; bN < blocks.size(); bN++) {
		if (blocks[bN].zone.type() == ZoneType::TYPE_PARK) continue;//skip those with parks

		for (int pi = 0; pi < blocks[bN].parcels.size(); ++pi) {
			if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_PARK) continue;
			if (blocks[bN].parcels[pi].myBuilding.buildingFootprint.contour.size() < 3) continue;

			float c = rand() % 192;
			blocks[bN].parcels[pi].myBuilding.color = QColor(c, c, c);
			if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_RESIDENTIAL) {
				if (blocks[bN].parcels[pi].zone.level() == 1) {
					PMBuildingHouse::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
				} else {
					PMBuildingTower::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
				}
			} else if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_COMMERCIAL) {
				blocks[bN].parcels[pi].myBuilding.subType = rand() % 2;
				PMBuildingRB::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
			} else if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_MANUFACTURING) {
				blocks[bN].parcels[pi].myBuilding.subType = rand() % 3;
				PMBuildingFactory::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
			} else if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_PUBLIC) {
				PMBuildingSchool::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
			} else if (blocks[bN].parcels[pi].zone.type() == ZoneType::TYPE_AMUSEMENT) {
				PMBuildingRB::generate(rendManager, "3d_building", blocks[bN].parcels[pi].myBuilding);
			}

			//int building_type = 1;//placeTypes.myPlaceTypes[blocks[bN].getMyPlaceTypeIdx()].getInt("building_type");
			//VBOGeoBuilding::generateBuilding(rendManager,blocks[bN].myParcels[*vi].myBuilding, building_type);				
		}
	}
	printf("Building generation is done.\n");

	return true;
}