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

bool MainWindow::savePreferences(std::vector<int>& user_ids, std::vector<std::vector<float> >& preferences, const QString& filename) {
	QFile file(filename);
 
	if (!file.open(QIODevice::WriteOnly)) return false;
	
	QTextStream out(&file);
	for (int i = 0; i < user_ids.size(); ++i) {
		out << user_ids[i] << "\t";
		for (int k = 0; k < preferences[i].size(); ++k) {
			if (k > 0) {
				out << ",";
			}
			out << preferences[i][k];
		}
		out << "\n";
	}

	file.close();

	return true;
}

void MainWindow::onLoadZoning() {
	/*
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
	*/
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
 * preference vectorをファイルから読み込み、それに基づいてベストのプランを計算する。
 */
void MainWindow::onBestPlan() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Load preference file..."), "", tr("Preference files (*.txt)"));
	if (filename.isEmpty()) return;
	
	QFile file(filename);
	file.open(QIODevice::ReadOnly);
 
	// preference vectorを読み込む
	std::vector<std::vector<float> > preferences;

	QTextStream in(&file);
	while (true) {
		QString str = in.readLine(0);
		if (str == NULL) break;

		QStringList preference_list = str.split("\t")[1].split(",");
		std::vector<float> preference;
		for (int i = 0; i < preference_list.size(); ++i) {
			preference.push_back(preference_list[i].toFloat());
		}

		preferences.push_back(preference);
	}
	
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preferences);

	// 3D更新
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	//VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
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
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
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
			msgBox.setText(client.reply());
			msgBox.exec();
			return;
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
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
	}
}

void MainWindow::onHCResults() {
	// feature一覧を取得
	HTTPClient client;
	client.setUrl("http://gnishida.site90.com/tasks.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
	}

	std::vector<std::pair<QString, QString> > tasks = JSON::parse(client.reply(), "tasks", "option1", "option2");

	// HC結果を取得
	client.setUrl("http://gnishida.site90.com/results.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
	}

	std::vector<std::pair<QString, QString> > results = JSON::parse(client.reply(), "results", "user_id", "choices");

	// compute preference vector using Gradient Descent
	std::vector<int> user_ids;
	std::vector<std::vector<float> > preferences;
	for (int u = 0; u < results.size(); ++u) {
		user_ids.push_back(results[u].first.toInt());
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

		printf("User: %d: ", user_ids[u]);
		for (int k = 0; k < w.size(); ++k) {
			printf("%lf,", w[k]);
		}
		printf("\n");
	}

	// current_roundを取得
	client.setUrl("http://gnishida.site90.com/get_current_round.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
	}
	int current_round = client.reply().toInt();
	QString filename = QString("preferences_%1.txt").arg(current_round);
	savePreferences(user_ids, preferences, filename);

	// ベストプランを計算する
	urbanGeometry->findBestPlan(glWidget->vboRenderManager, preferences);

	// 3D更新
	VBOPm::generateBlocks(glWidget->vboRenderManager, urbanGeometry->roads, urbanGeometry->blocks, urbanGeometry->zones);
	VBOPm::generateZoningMesh(glWidget->vboRenderManager, urbanGeometry->blocks);
	//VBOPm::generateParcels(glWidget->vboRenderManager, urbanGeometry->blocks);
	glWidget->shadow.makeShadowMap(glWidget);
	glWidget->updateGL();

	for (int u = 0; u < results.size(); ++u) {
		QImage img = generatePictureOfBestPlace(preferences[u]);
		QString filename = QString("%1_%2.png").arg(user_ids[u]).arg(current_round);
		img.save(filename);

		client.setUrl("http://gnishida.site90.com/upload.php");
		if (!client.uploadFile("upload.php", "file", filename, "image/png")) {
			QMessageBox msgBox(this);
			msgBox.setText(client.reply());
			msgBox.exec();
		}
	}
}

void MainWindow::onHCNext() {
	HTTPClient client;

	// max_stepを取得
	client.setUrl("http://gnishida.site90.com/get_max_step.php");
	if (!client.request()) {
		QMessageBox msgBox(this);
		msgBox.setText(client.reply());
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
			msgBox.setText(client.reply());
			msgBox.exec();
			return;
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
		msgBox.setText(client.reply());
		msgBox.exec();
		return;
	}
}

void MainWindow::onFileUpload() {
	HTTPClient client;
	client.setUrl("http://gnishida.site90.com/upload.php");
	if (client.uploadFile("upload.php", "file", "16.png", "image/png")) {
		QMessageBox msgBox(this);
		msgBox.setText("Server response: " + client.reply());
		msgBox.exec();
	}
}
