#include "MainWindow.h"
#include <QFileDialog>
#include "VBOPm.h"
#include "Util.h"
#include "VBOPmBlocks.h"
#include "VBOPmParcels.h"

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// setup the toolbar
	ui.fileToolBar->addAction(ui.actionLoadRoads);
	ui.fileToolBar->addAction(ui.actionSaveRoads);

	ui.actionModeDefault->setChecked(true);

	// register the menu's action handlers
	connect(ui.actionLoadZoning, SIGNAL(triggered()), this, SLOT(onLoadZoning()));
	connect(ui.actionLoadRoads, SIGNAL(triggered()), this, SLOT(onLoadRoads()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));

	connect(ui.actionGenerateBlocks, SIGNAL(triggered()), this, SLOT(onGenerateBlocks()));
	connect(ui.actionGenerateParcels, SIGNAL(triggered()), this, SLOT(onGenerateParcels()));
	connect(ui.actionGenerateBuildings, SIGNAL(triggered()), this, SLOT(onGenerateBuildings()));
	connect(ui.actionGenerateVegetation, SIGNAL(triggered()), this, SLOT(onGenerateVegetation()));
	connect(ui.actionGenerateAll, SIGNAL(triggered()), this, SLOT(onGenerateAll()));

	connect(ui.actionViewZoning, SIGNAL(triggered()), this, SLOT(onViewZoning()));
	connect(ui.actionPropose, SIGNAL(triggered()), this, SLOT(onPropose()));
	connect(ui.actionFindBest, SIGNAL(triggered()), this, SLOT(onFindBest()));
	connect(ui.actionCameraCar, SIGNAL(triggered()), this, SLOT(onCameraCar()));

	// setup the GL widget
	glWidget = new GLWidget3D(this);
	setCentralWidget(glWidget);

	urbanGeometry = new UrbanGeometry(this);

	//mode = MODE_DEFAULT;
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
	QString filename = QFileDialog::getOpenFileName(this, tr("Open zoning file..."), "", tr("Zoning Files (*.xml)"));
	if (filename.isEmpty()) return;

	urbanGeometry->zones.load(filename);

	// re-generate blocks
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);

	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);

	// re-generate parcels
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);

	urbanGeometry->allocateAll();
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);

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
	urbanGeometry->allocateAll();
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);
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
	urbanGeometry->allocateAll();
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);

	VBOPm::generateBuildings(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateVegetation(glWidget->vboRenderManager, urbanGeometry->blocks, urbanGeometry->zones);
	glWidget->shadow.makeShadowMap(glWidget);

	glWidget->updateGL();
}

void MainWindow::onViewZoning() {
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onPropose() {
	// randomly generate the zoning
	urbanGeometry->zones.generate(QVector2D(4000, 4000));

	// assign zone type to blocks
	VBOPmBlocks::assignZonesToBlocks(urbanGeometry->zones, urbanGeometry->blocks);

	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);

	// re-generate parcels
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);

	urbanGeometry->allocateAll();
	VBOPm::generatePeopleMesh(glWidget->vboRenderManager, urbanGeometry->people);

	float score = urbanGeometry->computeScore();
	printf("score: %lf\n", score);

	glWidget->updateGL();
}

void MainWindow::onFindBest() {
	float best1 = 0.0f;
	float best2 = 0.0f;
	float best3 = 0.0f;
	Zoning zoning1;
	Zoning zoning2;
	Zoning zoning3;

	for (int loop = 0; loop < 10; ++loop) {
		// randomly generate the zoning
		urbanGeometry->zones.generate(QVector2D(4000, 4000));

		// assign zone type to blocks
		VBOPmBlocks::assignZonesToBlocks(urbanGeometry->zones, urbanGeometry->blocks);

		// re-generate parcels
		VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);

		urbanGeometry->allocateAll();
		float score = urbanGeometry->computeScore();
		printf("score: %lf\n", score);

		if (score > best3) {
			best3 = score;
			zoning3 = urbanGeometry->zones;

			if (best3 > best2) {
				std::swap(best2, best3);
				std::swap(zoning2, zoning3);

				if (best2 > best1) {
					std::swap(best1, best2);
					std::swap(zoning1, zoning2);
				}
			}
		}
	}

	zoning1.save("zoning1.xml");
	zoning2.save("zoning2.xml");
	zoning3.save("zoning3.xml");
}

void MainWindow::onCameraCar() {
	//glWidget->camera = &glWidget->carCamera;
}
