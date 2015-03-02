#pragma once

#include <QWidget>
#include "ui_ExhaustiveSearchSetupWidget.h"

class ExhaustiveSearchSetupWidget : public QDialog {
Q_OBJECT

private:
	Ui::ExhaustiveSearchSetupWidget ui;

public:
	int gridSize;

public:
	ExhaustiveSearchSetupWidget(QWidget* parent);

public slots:
	void onOK();
};

