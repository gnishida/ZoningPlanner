#pragma once

#include <QDockWidget>
#include "ui_ControlWidget.h"
#include "Person.h"

class MainWindow;

class ControlWidget : public QDockWidget {
Q_OBJECT

private:
	MainWindow* mainWin;

public:
	Ui::ControlWidget ui;
	ControlWidget(MainWindow* mainWin);

	void showPersonInfo(const Person& person);

public slots:

};

