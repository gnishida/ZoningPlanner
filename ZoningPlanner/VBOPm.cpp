﻿/************************************************************************************************
 *		Procedural City Generation
 *		@author igarciad
 ************************************************************************************************/

#include "VBOPm.h"
#include "Polygon3D.h"

#include <qdir.h>
#include <QStringList>
#include <QTime>

#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"
#include "VBOPmBuildings.h"
#include "BlockSet.h"
#include "VBOGeoBuilding.h"
#include "VBOVegetation.h"
#include "Polygon3D.h"
#include "Util.h"
#include "HeatMapColorTable.h"
#include "PMBuildingHouse.h"
#include "PMBuildingTower.h"
#include "PMBuildingFactory.h"
#include "PMBuildingRB.h"
#include "PMBuildingSchool.h"

// LC
bool VBOPm::initializedLC=false;
static std::vector<QString> sideWalkFileNames;
static std::vector<QVector3D> sideWalkScale;
static std::vector<QString> grassFileNames;

/**
 * テクスチャ画像の読み込み
 */
void VBOPm::initLC(){
	QString pathName="../data/textures/LC";
	// 3. sidewalk
	QDir directorySW(pathName+"/sidewalk/");
	QStringList nameFilter;
	nameFilter << "*.png" << "*.jpg" << "*.gif";
	QStringList list = directorySW.entryList( nameFilter, QDir::Files );
	for(int lE=0;lE<list.size();lE++){
		if(QFile::exists(pathName+"/sidewalk/"+list[lE])){
			sideWalkFileNames.push_back(pathName+"/sidewalk/"+list[lE]);
			QStringList scaleS=list[lE].split("_");
			if(scaleS.size()!=4)
				sideWalkScale.push_back(QVector3D(1.0f,1.0f,0));
			else{
				sideWalkScale.push_back(QVector3D(scaleS[1].toFloat(),scaleS[2].toFloat(),0));
			}
		}
	}
	grassFileNames.push_back("../data/textures/LC/grass/grass01.jpg");
	grassFileNames.push_back("../data/textures/LC/grass/grass02.jpg");
	grassFileNames.push_back("../data/textures/LC/grass/grass03.jpg");
	grassFileNames.push_back("../data/textures/LC/grass/grass04.jpg");
	printf("-->Initialized LC\n");
	initializedLC=true;
}


/**
 * 道路網から、ブロック情報を抽出する。
 */
/*
bool VBOPm::generateBlocks(VBORenderManager& rendManager,RoadGraph &roadGraph, BlockSet& blocks, Zoning& zones){
	// INIT
	if(initializedLC==false){
		initLC();//init LC textures
	}

	if (!VBOPmBlocks::generateBlocks(zones, roadGraph,blocks)) {
		printf("ERROR: generateBlocks\n");
		return false;
	}

	generateBlockModels(rendManager, roadGraph, blocks);

	return true;
}
*/

/**
 * ブロック情報から、その3Dモデルを生成する
 */
/*
void VBOPm::generateBlockModels(VBORenderManager& rendManager,RoadGraph &roadGraph, BlockSet& blocks) {
	// 3Dモデルを生成する
	rendManager.removeStaticGeometry("3d_sidewalk");
	rendManager.removeStaticGeometry("3d_block");
	for (int i = 0; i < blocks.size(); ++i) {
		blocks[i].adaptToTerrain(&rendManager);

		// 歩道の3Dモデルを生成
		{
			int randSidewalk=2;//qrand()%grassFileNames.size();
			rendManager.addStaticGeometry2("3d_sidewalk", blocks[i].sidewalkContour.contour, 0.3f, false, sideWalkFileNames[randSidewalk], GL_QUADS, 2, sideWalkScale[randSidewalk], QColor());
			//sides
			std::vector<Vertex> vert;
			for (int sN = 0;sN < blocks[i].sidewalkContour.contour.size(); sN++) {
				int ind1 = sN;
				int ind2 = (sN+1) % blocks[i].sidewalkContour.contour.size();
				QVector3D dir = blocks[i].sidewalkContour.contour[ind2] - blocks[i].sidewalkContour.contour[ind1];
				float length = dir.length();
				dir /= length;
				//printf("z %f\n",blocks[bN].blockContour.contour[ind1].z());
				QVector3D p1 = blocks[i].sidewalkContour.contour[ind1]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p2 = blocks[i].sidewalkContour.contour[ind2]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p3 = blocks[i].sidewalkContour.contour[ind2]+QVector3D(0,0, 0.3f);//1.5f);
				QVector3D p4 = blocks[i].sidewalkContour.contour[ind1]+QVector3D(0,0, 0.3f);//1.5f);
				QVector3D normal = QVector3D::crossProduct(p2-p1,p4-p1).normalized();
				vert.push_back(Vertex(p1, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p2, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p3, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p4, QColor(128, 128, 128), normal, QVector3D()));
			}
			rendManager.addStaticGeometry("3d_sidewalk", vert, "", GL_QUADS, 1 | mode_Lighting);
		}

		// 公園の3Dモデルを生成
		if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {
			// PARK
			int randPark=qrand()%grassFileNames.size();
			rendManager.addStaticGeometry2("3d_sidewalk", blocks[i].blockContour.contour, 0.5f, false, grassFileNames[randPark], GL_QUADS, 2, QVector3D(0.05f,0.05f,0.05f), QColor());

			//sides
			std::vector<Vertex> vert;
			for (int sN = 0; sN < blocks[i].blockContour.contour.size(); sN++) {
				int ind1 = sN;
				int ind2 = (sN+1) % blocks[i].blockContour.contour.size();
				QVector3D dir = blocks[i].blockContour.contour[ind2] - blocks[i].blockContour.contour[ind1];
				float length = dir.length();
				dir /= length;
				//printf("z %f\n",blocks[bN].blockContour.contour[ind1].z());
				QVector3D p1 = blocks[i].blockContour.contour[ind1]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p2 = blocks[i].blockContour.contour[ind2]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p3 = blocks[i].blockContour.contour[ind2]+QVector3D(0,0, 0.5f);//1.5f);
				QVector3D p4 = blocks[i].blockContour.contour[ind1]+QVector3D(0,0, 0.5f);//1.5f);
				QVector3D normal = QVector3D::crossProduct(p2-p1,p4-p1).normalized();
				vert.push_back(Vertex(p1, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p2, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p3, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p4, QColor(128, 128, 128), normal, QVector3D()));
			}
			rendManager.addStaticGeometry("3d_sidewalk", vert, "", GL_QUADS, 1|mode_Lighting);
		}
	}
}
*/

/**
 * Block情報から、Parcel情報を計算する。
 */
/*
bool VBOPm::generateParcels(VBORenderManager& rendManager, BlockSet& blocks) {
	if (!VBOPmParcels::generateParcels(rendManager, blocks.blocks)) {
		printf("ERROR: generateParcels\n");
		return false;
	}

	generateParcelModels(rendManager, blocks);

	// ビルのfootprintを計算する
	if (!VBOPmBuildings::generateBuildings(rendManager, blocks.blocks)) {
		printf("ERROR: generateBuildings\n");
		return false;
	}
		
	return true;
}
*/

/**
 * Parcel情報から、その3Dモデルを生成する
 */
/*
void VBOPm::generateParcelModels(VBORenderManager& rendManager, BlockSet& blocks) {
	rendManager.removeStaticGeometry("3d_parcel");
	for (int i = 0; i < blocks.size(); ++i) {
		blocks[i].adaptToTerrain(&rendManager);

		Block::parcelGraphVertexIter vi, viEnd;
			
		int cnt = 0;
		for (boost::tie(vi, viEnd) = boost::vertices(blocks[i].myParcels); vi != viEnd; ++vi, ++cnt) {
			std::vector<Vertex> vert;
			QVector3D color;

			// 上面のモデル
			int randPark=1;//qrand()%grassFileNames.size();
			rendManager.addStaticGeometry2("3d_parcel",blocks[i].myParcels[*vi].parcelContour.contour,0.5f,false,grassFileNames[randPark],GL_QUADS,2,QVector3D(0.05f,0.05f,0.05f),QColor());

			// 側面のモデル
			for(int sN=0;sN<blocks[i].myParcels[*vi].parcelContour.contour.size();sN++){
				int ind1 = sN;
				int ind2 = (sN+1) % blocks[i].myParcels[*vi].parcelContour.contour.size();
				QVector3D dir = blocks[i].myParcels[*vi].parcelContour.contour[ind2] - blocks[i].myParcels[*vi].parcelContour.contour[ind1];
				float length = dir.length();
				dir /= length;
				
				QVector3D p1 = blocks[i].myParcels[*vi].parcelContour.contour[ind1]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p2 = blocks[i].myParcels[*vi].parcelContour.contour[ind2]+QVector3D(0,0, 0.0f);//1.0f);
				QVector3D p3 = blocks[i].myParcels[*vi].parcelContour.contour[ind2]+QVector3D(0,0, 0.5f);//1.5f);
				QVector3D p4 = blocks[i].myParcels[*vi].parcelContour.contour[ind1]+QVector3D(0,0, 0.5f);//1.5f);
				QVector3D normal = QVector3D::crossProduct(p2-p1,p4-p1).normalized();
				vert.push_back(Vertex(p1, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p4, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p3, QColor(128, 128, 128), normal, QVector3D()));
				vert.push_back(Vertex(p2, QColor(128, 128, 128), normal, QVector3D()));
			}
			rendManager.addStaticGeometry("3d_parcel",vert,"",GL_QUADS,1|mode_Lighting);
		}
	}
}
*/

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

void VBOPm::generatePeopleMesh(VBORenderManager& rendManager, std::vector<Person>& people) {
	// 3Dモデルを生成する
	rendManager.removeStaticGeometry("people");
	for (int i = 0; i < people.size(); ++i) {
		// 人の3Dモデルを生成
		{
			QColor color(255, 255, 255);
			rendManager.addSphere("people", QVector3D(people[i].homeLocation.x(), people[i].homeLocation.y(), 80), 5.0, color);
		}
	}
}

void VBOPm::generateSelecionMesh(VBORenderManager& rendManager, Person& person, UrbanGeometry& urbanGeometry) {
	// 3Dモデルを生成する
	rendManager.removeStaticGeometry("selection");
	rendManager.removeStaticGeometry("line");

	rendManager.addSphere("selection", QVector3D(person.homeLocation, 80), 10, QColor(255, 0, 0));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.stores[person.nearestStore].location, 80), 10, QColor(0, 0, 255));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.schools[person.nearestSchool].location, 80), 10, QColor(0, 192, 192));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.restaurants[person.nearestRestaurant].location, 80), 10, QColor(0, 0, 128));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.parks[person.nearestPark].location, 80), 10, QColor(0, 255, 0));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.amusements[person.nearestAmusement].location, 80), 10, QColor(255, 0, 255));
	rendManager.addSphere("selection", QVector3D(urbanGeometry.libraries[person.nearestLibrary].location, 80), 10, QColor(0, 255, 255));

	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.stores[person.nearestStore].location, 80), QColor(255, 255, 255));
	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.schools[person.nearestSchool].location, 80), QColor(255, 255, 255));
	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.restaurants[person.nearestRestaurant].location, 80), QColor(255, 255, 255));
	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.parks[person.nearestPark].location, 80), QColor(255, 255, 255));
	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.amusements[person.nearestAmusement].location, 80), QColor(255, 255, 255));
	rendManager.addLine("line", QVector3D(person.homeLocation, 80), QVector3D(urbanGeometry.libraries[person.nearestLibrary].location, 80), QColor(255, 255, 255));
}
