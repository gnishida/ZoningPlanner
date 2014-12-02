#include "MainWindow.h"
#include <QFileDialog>
#include "VBOPm.h"
#include "Util.h"
#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"
#include "ConvexHull.h"
#include <time.h>
#include <algorithm>
#include <QFile>
#include <QTextStream>
#include <iostream>
#include <fstream>
#include <string>

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// setup the docking widgets
	controlWidget = new ControlWidget(this);

	// setup the toolbar
	ui.fileToolBar->addAction(ui.actionLoadRoads);
	ui.fileToolBar->addAction(ui.actionSaveRoads);

	ui.actionModeDefault->setChecked(true);

	// register the menu's action handlers
	connect(ui.actionLoadZoning, SIGNAL(triggered()), this, SLOT(onLoadZoning()));
	connect(ui.actionLoadRoads, SIGNAL(triggered()), this, SLOT(onLoadRoads()));
	connect(ui.actionSaveImage, SIGNAL(triggered()), this, SLOT(onSaveImage()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));

	connect(ui.actionGenerateBlocks, SIGNAL(triggered()), this, SLOT(onGenerateBlocks()));
	connect(ui.actionGenerateParcels, SIGNAL(triggered()), this, SLOT(onGenerateParcels()));
	connect(ui.actionGenerateBuildings, SIGNAL(triggered()), this, SLOT(onGenerateBuildings()));
	connect(ui.actionGenerateVegetation, SIGNAL(triggered()), this, SLOT(onGenerateVegetation()));
	connect(ui.actionGenerateAll, SIGNAL(triggered()), this, SLOT(onGenerateAll()));

	connect(ui.actionViewGeometry, SIGNAL(triggered()), this, SLOT(onViewGeometry()));
	connect(ui.actionViewZoning, SIGNAL(triggered()), this, SLOT(onViewZoning()));
	connect(ui.actionViewPeople, SIGNAL(triggered()), this, SLOT(onViewPeople()));
	connect(ui.actionViewStore, SIGNAL(triggered()), this, SLOT(onViewStore()));
	connect(ui.actionViewSchool, SIGNAL(triggered()), this, SLOT(onViewSchool()));
	connect(ui.actionViewRestaurant, SIGNAL(triggered()), this, SLOT(onViewRestaurant()));
	connect(ui.actionViewPark, SIGNAL(triggered()), this, SLOT(onViewPark()));
	connect(ui.actionViewAmusement, SIGNAL(triggered()), this, SLOT(onViewAmusement()));
	connect(ui.actionViewLibrary, SIGNAL(triggered()), this, SLOT(onViewLibrary()));
	connect(ui.actionViewNoise, SIGNAL(triggered()), this, SLOT(onViewNoise()));
	connect(ui.actionViewPollution, SIGNAL(triggered()), this, SLOT(onViewPollution()));
	connect(ui.actionViewStation, SIGNAL(triggered()), this, SLOT(onViewStation()));

	connect(ui.actionPropose, SIGNAL(triggered()), this, SLOT(onPropose()));
	connect(ui.actionBestPlan, SIGNAL(triggered()), this, SLOT(onBestPlan()));
	connect(ui.actionBestPlanMT, SIGNAL(triggered()), this, SLOT(onBestPlanMT()));
	connect(ui.actionBestPlanGPU, SIGNAL(triggered()), this, SLOT(onBestPlanGPU()));
	connect(ui.actionBestPlanAndPeople, SIGNAL(triggered()), this, SLOT(onBestPlanAndPeople()));
	connect(ui.actionBestPlanAndPeopleMT, SIGNAL(triggered()), this, SLOT(onBestPlanAndPeopleMT()));
	connect(ui.actionBestPlanAndPeopleGPU, SIGNAL(triggered()), this, SLOT(onBestPlanAndPeopleGPU()));
	connect(ui.actionPeopleAllocation, SIGNAL(triggered()), this, SLOT(onPeopleAllocation()));

	// setup the GL widget
	glWidget = new GLWidget3D(this);
	setCentralWidget(glWidget);

	urbanGeometry = new UrbanGeometry(this);

	controlWidget->show();
	addDockWidget(Qt::LeftDockWidgetArea, controlWidget);
}

MainWindow::~MainWindow()
{

}

void MainWindow::keyPressEvent(QKeyEvent* e) {
	glWidget->keyPressEvent(e);
}

void MainWindow::keyReleaseEvent(QKeyEvent* e) {
	glWidget->keyReleaseEvent(e);
}

void MainWindow::onLoadZoning() {
	clock_t startTime, endTime;

	QString filename = QFileDialog::getOpenFileName(this, tr("Open zoning file..."), "", tr("Zoning Files (*.xml)"));
	if (filename.isEmpty()) return;

	startTime = clock();
	urbanGeometry->zones.load(filename);
	endTime = clock();
	printf("Load zone file: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	// re-generate blocks
	startTime = clock();
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	endTime = clock();
	printf("Blocks generation: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	startTime = clock();
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	endTime = clock();
	printf("Zoning mesh generation: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	// re-generate parcels
	startTime = clock();
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	endTime = clock();
	printf("Parcels generation: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	urbanGeometry->allocateAll();

	// レイヤー情報を更新する
	startTime = clock();
	urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);
	endTime = clock();
	printf("Layers generation: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	// 人のモデルを生成
	startTime = clock();
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);
	endTime = clock();
	printf("People model generation: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	// compute the feature vectors
	startTime = clock();
	urbanGeometry->computeScore();
	endTime = clock();
	printf("Compute score: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

	glWidget->shadow.makeShadowMap(glWidget);

	glWidget->updateGL();
}

void MainWindow::onLoadRoads() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Open Street Map file..."), "", tr("StreetMap Files (*.gsm)"));
	if (filename.isEmpty()) return;

	urbanGeometry->loadRoads(filename);
	glWidget->shadow.makeShadowMap(glWidget);

	glWidget->updateGL();
}

void MainWindow::onSaveImage() {
	if (!QDir("screenshots").exists()) QDir().mkdir("screenshots");
	QString fileName = "screenshots/" + QDate::currentDate().toString("yyMMdd") + "_" + QTime::currentTime().toString("HHmmss") + ".png";
	glWidget->grabFrameBuffer().save(fileName);
}

void MainWindow::onLoadCamera() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Open Camera file..."), "", tr("Area Files (*.cam)"));
	if (filename.isEmpty()) return;

	glWidget->camera->loadCameraPose(filename);
	glWidget->updateCamera();

	glWidget->updateGL();
}

void MainWindow::onSaveCamera() {
	QString filename = QFileDialog::getSaveFileName(this, tr("Save Camera file..."), "", tr("Area Files (*.cam)"));
	if (filename.isEmpty()) return;
	
	glWidget->camera->saveCameraPose(filename);
}

void MainWindow::onResetCamera() {
	glWidget->camera->resetCamera();
	glWidget->updateCamera();
	glWidget->updateGL();
}

void MainWindow::onGenerateBlocks() {
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	glWidget->updateGL();
}

void MainWindow::onGenerateParcels() {
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	//urbanGeometry->allocateAll();
	//VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);
	glWidget->updateGL();
}

void MainWindow::onGenerateBuildings() {
	VBOPm::generateBuildings(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateVegetation() {
	VBOPm::generateVegetation(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateAll() {
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);

	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	//urbanGeometry->allocateAll();
	//VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);

	VBOPm::generateBuildings(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateVegetation(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	glWidget->shadow.makeShadowMap(glWidget);

	glWidget->updateGL();
}

void MainWindow::onViewGeometry() {
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onViewZoning() {
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onViewPeople() {
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onViewStore() {
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewSchool() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewRestaurant() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewPark() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewAmusement() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewLibrary() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewNoise() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewPollution() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewStation->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onViewStation() {
	ui.actionViewStore->setChecked(false);
	ui.actionViewSchool->setChecked(false);
	ui.actionViewRestaurant->setChecked(false);
	ui.actionViewPark->setChecked(false);
	ui.actionViewAmusement->setChecked(false);
	ui.actionViewLibrary->setChecked(false);
	ui.actionViewNoise->setChecked(false);
	ui.actionViewPollution->setChecked(false);
	glWidget->updateGL();
}

void MainWindow::onPropose() {
	// randomly assign zone types to the blocks
	urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

	urbanGeometry->allocateAll();

	// generate 3D mesh
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);

	// レイヤー情報を更新する
	urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

	float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
	printf("score: %lf\n", score);

	glWidget->updateGL();
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlan() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	//srand(time(NULL));
	srand(0);

	std::vector<std::pair<float, Zoning> > zones;
	//for (int loop = 0; loop < 100; ++loop) {
	for (int loop = 0; loop < 3; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		time_t start = clock();
		urbanGeometry->allocateAll();
		time_t end = clock();
		std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
		urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		printf("%d: score=%lf\n", loop, score);

		zones.push_back(std::make_pair(score, urbanGeometry->zones));
	}

	// ベスト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoning());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoning());
		std::pair<float, Zoning> z = zones.back();

		if (i == 0) {
			urbanGeometry->zones = z.second;
		}

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}

	// ワースト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoningReverse());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoningReverse());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlanMT() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	//srand(time(NULL));
	srand(0);

	std::vector<std::pair<float, Zoning> > zones;
	for (int loop = 0; loop < 100; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		time_t start = clock();
		urbanGeometry->allocateAll();
		time_t end = clock();
		std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
		urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		printf("%d: score=%lf\n", loop, score);

		zones.push_back(std::make_pair(score, urbanGeometry->zones));
	}

	// ベスト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoning());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoning());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}

	// ワースト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoningReverse());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoningReverse());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlanGPU() {
	printf("onBestPlanGPU\n");

	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	urbanGeometry->findBestPlanGPU(glWidget->vboRenderManager, 1000);
	printf("onBestPlanGPU done\n");
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 各プランに対して、人の配置を最適化して、スコアを改善する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlanAndPeople() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	srand(time(NULL));

	// 時間計測開始
	time_t start = clock();

	std::vector<std::pair<float, Zoning> > zones;
	for (int loop = 0; loop < 10; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		urbanGeometry->allocateAll();

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
		urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

		// 人を動かす
		urbanGeometry->movePeople(glWidget->vboRenderManager);

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		printf("%d: score=%lf\n", loop, score);

		zones.push_back(std::make_pair(score, urbanGeometry->zones));
	}

	// 時間計測終了
	time_t end = clock();
	std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

	// ベスト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoning());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoning());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}

	// ワースト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoningReverse());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoningReverse());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 各プランに対して、人の配置を最適化して、スコアを改善する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlanAndPeopleMT() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	srand(time(NULL));

	std::vector<std::pair<float, Zoning> > zones;
	for (int loop = 0; loop < 10; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		urbanGeometry->allocateAll();

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
		urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

		// 人を動かす
		time_t start = clock();
		urbanGeometry->movePeopleMT(glWidget->vboRenderManager);
		time_t end = clock();
		std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		printf("%d: score=%lf\n", loop, score);

		zones.push_back(std::make_pair(score, urbanGeometry->zones));
	}

	// ベスト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoning());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoning());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}

	// ワースト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoningReverse());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoningReverse());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}
}

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 各プランに対して、人の配置を最適化して、スコアを改善する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlanAndPeopleGPU() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	srand(time(NULL));

	std::vector<std::pair<float, Zoning> > zones;
	for (int loop = 0; loop < 10; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		urbanGeometry->allocateAll();

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
		urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

		// 人を動かす
		time_t start = clock();
		urbanGeometry->movePeopleGPU(glWidget->vboRenderManager);
		time_t end = clock();
		std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		printf("%d: score=%lf\n", loop, score);

		zones.push_back(std::make_pair(score, urbanGeometry->zones));
	}

	// ベスト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoning());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoning());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}

	// ワースト３を保存する
	std::make_heap(zones.begin(), zones.end(), CompareZoningReverse());
	for (int i = 0; i < 3; ++i) {
		std::pop_heap(zones.begin(), zones.end(), CompareZoningReverse());
		std::pair<float, Zoning> z = zones.back();

		QString filename = QString("zoning/score_%1.xml").arg(z.first, 4, 'f', 6);
		z.second.save(filename);

		zones.pop_back();
	}
}

/**
 * 現在のゾーンに対して、人をランダムに配置し、スコアを計算する。これをＮ回繰り返して、スコアリストをファイルに出力する。
 * Pythonプログラムなどでヒストグラム生成に使用できる。
 */
void MainWindow::onPeopleAllocation() {
	urbanGeometry->allocateAll();

	// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
	urbanGeometry->updateLayer(0, glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateLayer(1, glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateLayer(2, glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateLayer(3, glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateLayer(4, glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLayer(5, glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateLayer(6, glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updateLayer(7, glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateLayer(8, glWidget->vboRenderManager.vboStationLayer);

	std::vector<float> scores;
	for (int iter = 0; iter < 1000; ++iter) {
		// 人を動かす
		urbanGeometry->allocatePeople();

		float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
		scores.push_back(score);
		printf("%d: score=%lf\n", iter, score);
	}

	// スコアリストをファイルに保存
	std::ofstream out("score.txt");
	for (int i = 0; i < scores.size(); ++i) {
		out << scores[i] << std::endl;
	}
	out.close();
}

