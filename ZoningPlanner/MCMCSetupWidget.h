#pragma once

#include <QWidget>
#include "ui_MCMCSetupWidget.h"

class MCMCSetupWidget : public QDialog {
Q_OBJECT

private:
	Ui::MCMCSetupWidget ui;

public:
	int initialGridSize;
	int numStages;
	int MCMCSteps;
	float upscaleFactor;

public:
	MCMCSetupWidget(QWidget* parent);

public slots:
	void onOK();
};

