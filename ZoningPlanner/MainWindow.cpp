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
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QMessageBox>
#include "HTTPClient.h"
#include "HCStartWidget.h"
#include "JSON.h"
#include "GradientDescent.h"
#include "MCMC.h"

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

	connect(ui.actionBestPlan, SIGNAL(triggered()), this, SLOT(onBestPlan()));
	connect(ui.actionCameraDefault, SIGNAL(triggered()), this, SLOT(onCameraDefault()));

	connect(ui.actionHCStart, SIGNAL(triggered()), this, SLOT(onHCStart()));
	connect(ui.actionHCResults, SIGNAL(triggered()), this, SLOT(onHCResults()));
	connect(ui.actionHCNext, SIGNAL(triggered()), this, SLOT(onHCNext()));

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

QImage MainWindow::generatePictureOfBestPlace(std::vector<float>& preference) {
	QVector3D pt = QVector3D(urbanGeometry->findBestPlace(glWidget->vboRenderManager, preference));

	glWidget->camera2D.setTranslation(0, 0, 200.0f);
	glWidget->camera2D.setLookAt(pt.x(), pt.y(), 70);
	glWidget->camera2D.setXRotation(-60);
	glWidget->camera2D.setZRotation(-Util::rad2deg(atan2f(pt.x(), -pt.y())));
	glWidget->camera2D.updateCamMatrix();

	glWidget->updateGL();
	return glWidget->grabFrameBuffer();
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

/**
 * ランダムにプランを生成し、ランダムに人などを配備してそのスコアを決定する。
 * 一定回数繰り返して、ベスト３とワースト３のプランを保存する。
 */
void MainWindow::onBestPlan() {
	std::vector<std::vector<float> > preference;

	preference.resize(10);
	for (int i = 0; i < 10; ++i) preference[i].resize(9);

	//preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0; preference[0][3] = 0; preference[0][4] = 0; preference[0][5] = 0; preference[0][6] = 0; preference[0][7] = 1.0;
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0.2; preference[0][3] = 0.2; preference[0][4] = 0.2; preference[0][5] = 0; preference[0][6] = 0.1; preference[0][7] = 0.3;
	preference[1][0] = 0; preference[1][1] = 0; preference[1][2] = 0.15; preference[1][3] = 0; preference[1][4] = 0.45; preference[1][5] = 0; preference[1][6] = 0.2; preference[1][7] = 0.2;
	preference[2][0] = 0; preference[2][1] = 0; preference[2][2] = 0.1; preference[2][3] = 0; preference[2][4] = 0; preference[2][5] = 0; preference[2][6] = 0.4; preference[2][7] = 0.5;
	preference[3][0] = 0.15; preference[3][1] = 0.13; preference[3][2] = 0; preference[3][3] = 0.14; preference[3][4] = 0; preference[3][5] = 0.08; preference[3][6] = 0.2; preference[3][7] = 0.3;
	preference[4][0] = 0.3; preference[4][1] = 0; preference[4][2] = 0.3; preference[4][3] = 0.1; preference[4][4] = 0; preference[4][5] = 0; preference[4][6] = 0.1; preference[4][7] = 0.2;
	preference[5][0] = 0.05; preference[5][1] = 0; preference[5][2] = 0.15; preference[5][3] = 0.2; preference[5][4] = 0.15; preference[5][5] = 0; preference[5][6] = 0.15; preference[5][7] = 0.3;
	preference[6][0] = 0.2; preference[6][1] = 0.1; preference[6][2] = 0; preference[6][3] = 0.2; preference[6][4] = 0; preference[6][5] = 0.1; preference[6][6] = 0.1; preference[6][7] = 0.3;
	preference[7][0] = 0.3; preference[7][1] = 0; preference[7][2] = 0.3; preference[7][3] = 0; preference[7][4] = 0.2; preference[7][5] = 0; preference[7][6] = 0.1; preference[7][7] = 0.1;
	preference[8][0] = 0.25; preference[8][1] = 0; preference[8][2] = 0.1; preference[8][3] = 0.05; preference[8][4] = 0; preference[8][5] = 0; preference[8][6] = 0.25; preference[8][7] = 0.35;
	preference[9][0] = 0.25; preference[9][1] = 0; preference[9][2] = 0.2; preference[9][3] = 0; preference[9][4] = 0; preference[9][5] = 0; preference[9][6] = 0.2; preference[9][7] = 0.35;
	
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preference);

	// 3D更新
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();
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
	glWidget->camera2D.updateCamMatrix();

	glWidget->updateGL();
}

void MainWindow::onHCStart() {
	HCStartWidget dlg(this);
	if (dlg.exec() != QDialog::Accepted) {
		return;
	}

	// 適当なpreference vectorを作成
	std::vector<std::vector<float> > preference;
	preference.resize(1);
	for (int i = 0; i < 1; ++i) preference[i].resize(7);
	preference[0][0] = 0.378; preference[0][1] = 0.378; preference[0][2] = 0.378; preference[0][3] = 0.378; preference[0][4] = 0.378; preference[0][5] = 0.378; preference[0][6] = -0.378;

	// ゾーンプランを作成する
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preference);

	// HC初期化
	HTTPClient client;
	int max_round = dlg.max_round;
	int max_step = dlg.max_step;
	QString url = QString("http://gnishida.site90.com/config.php?current_round=0&max_round=%1&max_step=%1").arg(max_round).arg(max_step);
	client.setUrl(url);
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
	}

	// HCタスク生成
	std::vector<std::pair<std::vector<float>, std::vector<float> > > tasks = urbanGeometry->generateTasks(max_step);

	// HCタスクをアップロード
	for (int step = 0; step < max_step; ++step) {
		QString option1 = Util::join(tasks[step].first, ",");
		QString option2 = Util::join(tasks[step].second, ",");
		QString url = QString("http://gnishida.site90.com/add_task.php?step=%1&option1=%2&option2=%3").arg(step + 1).arg(option1).arg(option2);
		client.setUrl(url);
		if (!client.request()) {
			QMessageBox msgBox(this);
			msgBox.setText(client.error());
			msgBox.exec();
			break;
		}
	}

	// HCラウンドを1にセット
	client.setUrl("http://gnishida.site90.com/next_round.php");
	if (client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText("Server response: " + client.reply());
		msgBox.exec();
	} else {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
	}
}

void MainWindow::onHCResults() {
	// feature一覧を取得
	HTTPClient client;
	client.setUrl("http://gnishida.site90.com/tasks.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
		return;
	}

	std::vector<std::pair<QString, QString> > tasks = JSON::parse(client.reply(), "tasks", "option1", "option2");

	// HC結果を取得
	client.setUrl("http://gnishida.site90.com/results.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
		return;
	}

	std::vector<std::pair<QString, QString> > results = JSON::parse(client.reply(), "results", "user_id", "choices");

	// compute preference vector using Gradient Descent
	std::vector<std::vector<float> > preferences;
	for (int u = 0; u < results.size(); ++u) {
		printf("%s : %s\n", results[u].first.toUtf8().data(), results[u].second.toUtf8().data());

		QStringList chioces_list = results[u].second.split(",");

		GradientDescent gd;
		std::vector<std::pair<std::vector<float>, std::vector<float> > > features;
		std::vector<int> choices;

		for (int step = 0; step < tasks.size(); ++step) {
			std::vector<float> f1;
			std::vector<float> f2;
			QStringList feature1_list = tasks[step].first.split(",");
			QStringList feature2_list = tasks[step].second.split(",");
			for (int k = 0; k < 7; ++k) {
				f1.push_back(MCMC::distToFeature(feature1_list[k].toFloat()));
				f2.push_back(MCMC::distToFeature(feature2_list[k].toFloat()));
			}

			features.push_back(std::make_pair(f1, f2));

			choices.push_back(chioces_list[step].toInt() == 1 ? 1 : 0);
		}

		std::vector<float> w(7);
		w[0] = 0.3f; w[1] = 0.3f; w[2] = 0.3f; w[3] = 0.3f; w[4] = 0.3f; w[5] = 0.3f; w[6] = -0.3f;
		gd.run(w, features, choices, 10000, false, 0.0, 0.001, 0.0001);
		preferences.push_back(w);

		printf("User: %s: ", results[u].first.toUtf8().data());
		for (int k = 0; k < w.size(); ++k) {
			printf("%lf,", w[k]);
		}
		printf("\n");
	}

	// ベストプランを計算する
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preferences);

	// 3D更新
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();

	for (int u = 0; u < results.size(); ++u) {
		QImage img = generatePictureOfBestPlace(preferences[u]);
		QString filename = QString("bestplace_%1.png").arg(results[u].first);
		img.save(filename);
	}
}

void MainWindow::onHCNext() {
	HTTPClient client;

	// max_stepを取得
	client.setUrl("http://gnishida.site90.com/get_max_step.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
		return;
	}
	int max_step = client.reply().toInt();

	// HCタスク生成
	std::vector<std::pair<std::vector<float>, std::vector<float> > > tasks = urbanGeometry->generateTasks(max_step);

	// HCタスクをアップロード
	for (int step = 0; step < max_step; ++step) {
		QString option1 = Util::join(tasks[step].first, ",");
		QString option2 = Util::join(tasks[step].second, ",");
		QString url = QString("http://gnishida.site90.com/add_task.php?step=%1&option1=%2&option2=%3").arg(step + 1).arg(option1).arg(option2);
		client.setUrl(url);
		if (!client.request()) {
			QMessageBox msgBox(this);
			msgBox.setText(client.error());
			msgBox.exec();
			break;
		}
	}

	// HCラウンドをインクリメント
	client.setUrl("http://gnishida.site90.com/next_round.php");
	if (client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText("Server response: " + client.reply());
		msgBox.exec();
	} else {
		QMessageBox msgBox(this);
		msgBox.setText(client.error());
		msgBox.exec();
	}
}
