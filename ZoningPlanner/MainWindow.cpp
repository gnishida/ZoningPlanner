#include "MainWindow.h"
#include <QFileDialog>
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
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QMessageBox>
#include "HTTPClient.h"
#include "MCMCSetupWidget.h"
#include "ExhaustiveSearchSetupWidget.h"
#include "HCStartWidget.h"
#include "JSON.h"
#include "GradientDescent.h"
#include "MCMC5.h"
#include "MCMCUtil.h"
#include "BrushFire.h"
#include "HumanComputation.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags) : QMainWindow(parent, flags) {
	ui.setupUi(this);

	// setup the docking widgets
	controlWidget = new ControlWidget(this);

	// setup the toolbar
	ui.fileToolBar->addAction(ui.actionLoadRoads);
	ui.fileToolBar->addAction(ui.actionSaveRoads);

	ui.actionModeDefault->setChecked(true);

	// register the menu's action handlers
	connect(ui.actionLoadZoning, SIGNAL(triggered()), this, SLOT(onLoadZoning()));
	connect(ui.actionSaveZoning, SIGNAL(triggered()), this, SLOT(onSaveZoning()));
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

	connect(ui.actionBestPlan, SIGNAL(triggered()), this, SLOT(onBestPlan()));
	connect(ui.actionExhaustiveSearch, SIGNAL(triggered()), this, SLOT(onExhaustiveSearch()));
	connect(ui.actionScoreTest, SIGNAL(triggered()), this, SLOT(onScoreTest()));

	connect(ui.actionCameraDefault, SIGNAL(triggered()), this, SLOT(onCameraDefault()));
	connect(ui.actionCameraTest, SIGNAL(triggered()), this, SLOT(onCameraTest()));

	connect(ui.actionHCStart, SIGNAL(triggered()), this, SLOT(onHCStart()));
	connect(ui.actionHCResults, SIGNAL(triggered()), this, SLOT(onHCResults()));
	connect(ui.actionHCNext, SIGNAL(triggered()), this, SLOT(onHCNext()));
	connect(ui.actionFileUpload, SIGNAL(triggered()), this, SLOT(onFileUpload()));

	// setup the GL widget
	glWidget = new GLWidget3D(this);
	setCentralWidget(glWidget);

	urbanGeometry = new UrbanGeometry(this);
	urbanGeometry->loadInitZones("init_zones.xml");

	controlWidget->show();
	addDockWidget(Qt::LeftDockWidgetArea, controlWidget);
}

MainWindow::~MainWindow() {
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
	glWidget->keyPressEvent(e);
}

void MainWindow::keyReleaseEvent(QKeyEvent* e) {
	glWidget->keyReleaseEvent(e);
}

QImage MainWindow::generatePictureOfBestPlace(std::vector<float>& preference) {
	QVector2D pt = urbanGeometry->findBestPlace(glWidget->vboRenderManager, preference);
	std::cout << pt.x() << "," << pt.y() << std::endl;

	ui.actionViewZoning->setChecked(true);

	return glWidget->generatePictureOfPointInterest(pt);
}

bool MainWindow::savePreferences(QMap<int, std::vector<float> >& preferences, const QString& filename) {
	QFile file(filename);
 
	if (!file.open(QIODevice::WriteOnly)) return false;
	
	QTextStream out(&file);
	for (QMap<int, std::vector<float> >::iterator it = preferences.begin(); it != preferences.end(); ++it) {
		out << it.key() << "\t";
		for (int k = 0; k < it.value().size(); ++k) {
			if (k > 0) {
				out << ",";
			}
			out << it.value()[k];
		}
		out << "\n";
	}

	file.close();

	return true;
}

void MainWindow::onLoadZoning() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Open zoning file..."), "", tr("Zoning Files (*.xml)"));
	if (filename.isEmpty()) return;

	urbanGeometry->zones.load(filename);

	// re-generate blocks
	urbanGeometry->generateBlocks();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onSaveZoning() {
	QString filename = QFileDialog::getSaveFileName(this, tr("Save zoning file..."), "", tr("Zoning Files (*.xml)"));
	if (filename.isEmpty()) return;

	urbanGeometry->zones.save(filename);
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
	urbanGeometry->generateBlocks();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateParcels() {
	urbanGeometry->generateParcels();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateBuildings() {
	urbanGeometry->generateBuildings();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateVegetation() {
	urbanGeometry->generateVegetation();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onGenerateAll() {
	urbanGeometry->generateAll();
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

/**
 * preference vectorをファイルから読み込み、それに基づいてベストのプランを計算する。
 */
void MainWindow::onBestPlan() {
	MCMCSetupWidget dlg(this);
	if (dlg.exec() != QDialog::Accepted) {
		return;
	}

	QString filename = QFileDialog::getOpenFileName(this, tr("Load preference file..."), "", tr("Preference files (*.txt)"));
	if (filename.isEmpty()) return;
	
	// preference vectorを読み込む
	std::vector<std::vector<float> > preferences = mcmcutil::MCMCUtil::readPreferences(filename);
	
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preferences, dlg.initialGridSize, dlg.numStages, dlg.MCMCSteps, dlg.upscaleFactor, 10.0f);

	// 3D更新
	urbanGeometry->generateBlocks();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
}

void MainWindow::onExhaustiveSearch() {
	ExhaustiveSearchSetupWidget dlg(this);
	if (dlg.exec() != QDialog::Accepted) {
		return;
	}

	QString filename = QFileDialog::getOpenFileName(this, tr("Load preference file..."), "", tr("Preference files (*.txt)"));
	if (filename.isEmpty()) return;
	 
	// preference vectorを読み込む
	std::vector<std::vector<float> > preferences = mcmcutil::MCMCUtil::readPreferences(filename);

	// ゾーンプランを作成する
	urbanGeometry->findOptimalPlan(glWidget->vboRenderManager, preferences, dlg.gridSize);

}

void MainWindow::onScoreTest() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Load preference file..."), "", tr("Preference files (*.txt)"));
	if (filename.isEmpty()) return;
	
	// preference vectorを読み込む
	std::vector<std::vector<float> > preferences = mcmcutil::MCMCUtil::readPreferences(filename);

	filename = QFileDialog::getOpenFileName(this, tr("Load zone file..."), "", tr("Zone files (*.txt)"));
	if (filename.isEmpty()) return;
	
	std::vector<uchar> zones = mcmcutil::MCMCUtil::readZone(filename);
	int city_size = sqrtf(zones.size());

	brushfire::BrushFire bf(city_size, city_size, Zoning::NUM_COMPONENTS, zones);
	
	//float score = mcmcutil::MCMCUtil::computeScore(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);
	float score = mcmcutil::MCMCUtil::computeScoreCUDA(city_size, Zoning::NUM_COMPONENTS, bf.zones(), bf.distMap(), preferences);

	static float best_score = -100;
	if (score > best_score) {
		best_score = score;
	}

	printf("Score: %lf\n", score);
}

/**
 * 当該ユーザのpreference vectorに基づいて、カメラ位置を決定する。
 */
void MainWindow::onCameraDefault() {
	std::vector<float> preference;
	preference.resize(9);
	preference[0] = 0; preference[1] = 0; preference[2] = 0.2; preference[3] = 0.2; preference[4] = 0.2; preference[5] = 0; preference[6] = 0.1; preference[7] = 0.3;

	QVector3D pt = QVector3D(urbanGeometry->findBestPlace(glWidget->vboRenderManager, preference));

	glWidget->camera2D.setTranslation(0, 0, 200.0f);
	glWidget->camera2D.setLookAt(pt.x(), pt.y(), 70);
	glWidget->camera2D.setXRotation(-60);
	glWidget->camera2D.setZRotation(-Util::rad2deg(atan2f(pt.x(), -pt.y())));
	glWidget->updateCamera();

	glWidget->updateGL();
}

void MainWindow::onCameraTest() {
	QVector2D pt(500, -500);

	QImage img = glWidget->generatePictureOfPointInterest(pt);
}

void MainWindow::onHCStart() {
	HCStartWidget dlg(this);
	if (dlg.exec() != QDialog::Accepted) {
		return;
	}

	// 適当なpreference vectorを作成
	std::vector<std::vector<float> > preference(1, std::vector<float>(5));
	preference[0][0] = 0.378; preference[0][1] = -0.378; preference[0][2] = 0.378; preference[0][3] = -0.378; preference[0][4] = 0.378;

	// ゾーンプランを作成する
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preference, 4, 5, 200000, 1.0, 10.0f);

	// HC初期化
	HumanComputation hc;
	hc.init(dlg.max_round, dlg.max_step);

	// HCタスク生成
	std::vector<std::pair<std::vector<float>, std::vector<float> > > tasks = urbanGeometry->generateTasks(glWidget->vboRenderManager, dlg.max_step);

	// HCタスクをアップロード
	hc.uploadTasks(tasks);

	// 3D更新
	urbanGeometry->generateBlocks();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();

	// HCラウンドを1にセット
	hc.nextRound();
}

void MainWindow::onHCResults() {
	HumanComputation hc;

	// compute preference vector using Gradient Descent
	QMap<int, std::vector<float> > preferences = hc.computePreferences();


	// current_roundを取得
	int current_round = hc.getCurrentRound();

	QString filename = QString("preferences_%1.txt").arg(current_round);
	savePreferences(preferences, filename);

	std::vector<std::vector<float> > preferences2(preferences.values().size());
	for (int i = 0; i < preferences.values().size(); ++i) {
		preferences2[i].resize(preferences[i].size());
		for (int j = 0; j < preferences[i].size(); ++j) {
			preferences2[i][j] = preferences.values()[i][j];
		}
	}

	// ベストプランを計算する
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preferences2, 5, 5, 20000, 1.0, 10.0f);

	// 3D更新
	urbanGeometry->generateBlocks();
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();

	for (QMap<int, std::vector<float> >::iterator it = preferences.begin(); it != preferences.end(); ++it) {
		QImage img = generatePictureOfBestPlace(it.value());
		QString filename = QString("%1_%2.png").arg(it.key()).arg(current_round);
		img.save(filename);

		hc.uploadImage(filename);
	}
}

void MainWindow::onHCNext() {
	HumanComputation hc;

	int max_step = hc.getMaxStep();

	// HCタスク生成
	std::vector<std::pair<std::vector<float>, std::vector<float> > > tasks = urbanGeometry->generateTasks(glWidget->vboRenderManager, max_step);

	// HCタスクをアップロード
	hc.uploadTasks(tasks);

	// HCラウンドをインクリメント
	try {
		hc.nextRound();
	} catch (const QString& err) {
		QMessageBox msgBox(this);
		msgBox.setText(err);
		msgBox.exec();
		return;
	}
}

void MainWindow::onFileUpload() {
	HumanComputation hc;

	try {
		hc.uploadImage("16.png");
	} catch (const QString& err) {
		QMessageBox msgBox(this);
		msgBox.setText("Server response: " + err);
		msgBox.exec();
	}
}

