#pragma once

#include <QWidget>
#include "ui_HCStartWidget.h"

class HCStartWidget : public QDialog {
Q_OBJECT

private:
	Ui::HCStartWidget ui;

public:
	int max_round;
	int max_step;

public:
	HCStartWidget(QWidget* parent);

public slots:
	void onOK();
};

