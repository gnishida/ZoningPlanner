#include "MainWindow.h"
#include <QFileDialog>
#include "VBOPm.h"
#include "Util.h"
#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"
#include "ConvexHull.h"
#include <time.h>
#include <algorithm>

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
	connect(ui.actionFindBest, SIGNAL(triggered()), this, SLOT(onFindBest()));
	connect(ui.actionCameraCar, SIGNAL(triggered()), this, SLOT(onCameraCar()));

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
	urbanGeometry->updateStoreMap(glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateSchoolMap(glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateRestaurantMap(glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateParkMap(glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateAmusementMap(glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLibraryMap(glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateNoiseMap(glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updatePollutionMap(glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateStationMap(glWidget->vboRenderManager.vboStationLayer);
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
	urbanGeometry->updateStoreMap(glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateSchoolMap(glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateRestaurantMap(glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateParkMap(glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateAmusementMap(glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLibraryMap(glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateNoiseMap(glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updatePollutionMap(glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateStationMap(glWidget->vboRenderManager.vboStationLayer);

	float score = urbanGeometry->computeScore(glWidget->vboRenderManager);
	printf("score: %lf\n", score);

	glWidget->updateGL();
}

void MainWindow::onFindBest() {
	// generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	srand(time(NULL));

	std::vector<std::pair<float, Zoning> > zones;
	for (int loop = 0; loop < 1000; ++loop) {
		// randomly assign zone types to the blocks
		urbanGeometry->zones.randomlyAssignZoneType(urbanGeometry->blocks);

		urbanGeometry->allocateAll();

		// 各ブロックのゾーンタイプに基づき、レイヤー情報を更新する
 		urbanGeometry->updateStoreMap(glWidget->vboRenderManager.vboStoreLayer);
		urbanGeometry->updateSchoolMap(glWidget->vboRenderManager.vboSchoolLayer);
		urbanGeometry->updateRestaurantMap(glWidget->vboRenderManager.vboRestaurantLayer);
		urbanGeometry->updateParkMap(glWidget->vboRenderManager.vboParkLayer);
		urbanGeometry->updateAmusementMap(glWidget->vboRenderManager.vboAmusementLayer);
		urbanGeometry->updateLibraryMap(glWidget->vboRenderManager.vboLibraryLayer);
		urbanGeometry->updateNoiseMap(glWidget->vboRenderManager.vboNoiseLayer);
		urbanGeometry->updatePollutionMap(glWidget->vboRenderManager.vboPollutionLayer);
		urbanGeometry->updateStationMap(glWidget->vboRenderManager.vboStationLayer);

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

	// レイヤー情報を更新する
	/*
	urbanGeometry->updateStoreMap(glWidget->vboRenderManager.vboStoreLayer);
	urbanGeometry->updateSchoolMap(glWidget->vboRenderManager.vboSchoolLayer);
	urbanGeometry->updateRestaurantMap(glWidget->vboRenderManager.vboRestaurantLayer);
	urbanGeometry->updateParkMap(glWidget->vboRenderManager.vboParkLayer);
	urbanGeometry->updateAmusementMap(glWidget->vboRenderManager.vboAmusementLayer);
	urbanGeometry->updateLibraryMap(glWidget->vboRenderManager.vboLibraryLayer);
	urbanGeometry->updateNoiseMap(glWidget->vboRenderManager.vboNoiseLayer);
	urbanGeometry->updatePollutionMap(glWidget->vboRenderManager.vboPollutionLayer);
	urbanGeometry->updateStationMap(glWidget->vboRenderManager.vboStationLayer);
	*/
}

void MainWindow::onCameraCar() {
	//glWidget->camera = &glWidget->carCamera;
}
