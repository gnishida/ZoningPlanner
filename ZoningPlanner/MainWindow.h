#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include "ui_MainWindow.h"
#include "GLWidget3D.h"
#include "UrbanGeometry.h"
#include "ControlWidget.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
	~MainWindow();

public:
	Ui::MainWindow ui;
	ControlWidget* controlWidget;
	GLWidget3D* glWidget;
	UrbanGeometry* urbanGeometry;

protected:
	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);

public:
	QImage generatePictureOfBestPlace(std::vector<float>& preference);
	bool savePreferences(std::vector<int>& user_ids, std::vector<std::vector<float> >& preferences, const QString& filename);

public slots:
	void onLoadZoning();
	void onSaveZoning();
	void onLoadRoads();
	void onSaveImage();
	void onLoadCamera();
	void onSaveCamera();
	void onResetCamera();
	void onGenerateBlocks();
	void onGenerateParcels();
	void onGenerateBuildings();
	void onGenerateVegetation();
	void onGenerateAll();
	void onViewGeometry();
	void onViewZoning();
	void onBestPlan();
	void onExhaustiveSearch();
	void onCameraDefault();
	void onCameraTest();
	void onHCStart();
	void onHCResults();
	void onHCNext();
	void onFileUpload();
};

#endif // MAINWINDOW_H
