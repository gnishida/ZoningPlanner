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

public slots:
	void onLoadZoning();
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
	void onViewPeople();
	void onViewStore();
	void onViewSchool();
	void onViewRestaurant();
	void onViewPark();
	void onViewAmusement();
	void onViewLibrary();
	void onViewNoise();
	void onViewPollution();
	void onViewStation();
	void onPropose();
	void onBestPlan();
	void onBestPlanAndPeople();
	void onPeopleAllocation();
};

#endif // MAINWINDOW_H
