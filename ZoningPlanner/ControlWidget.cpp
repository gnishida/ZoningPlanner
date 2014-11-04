﻿#include "ControlWidget.h"
#include <QFileDialog>
#include "MainWindow.h"
#include "GLWidget3D.h"
#include "global.h"

ControlWidget::ControlWidget(MainWindow* mainWin) : QDockWidget("Control Widget", (QWidget*)mainWin) {
	this->mainWin = mainWin;

	// set up the UI
	ui.setupUi(this);

	hide();	
}

void ControlWidget::showPersonInfo(const Person& person) {
	switch (person.type()) {
	case Person::TYPE_UNKNOWN:
		ui.lineEditPersonType->setText("---");
		break;
	case Person::TYPE_STUDENT:
		ui.lineEditPersonType->setText("Student");
		break;
	case Person::TYPE_HOUSEWIFE:
		ui.lineEditPersonType->setText("Housewife");
		break;
	case Person::TYPE_OFFICEWORKER:
		ui.lineEditPersonType->setText("Office worker");
		break;
	case Person::TYPE_ELDERLY:
		ui.lineEditPersonType->setText("Elderly");
		break;
	}

	ui.lineEditPreference0->setText(QString::number(person.preference[0]));
	ui.lineEditPreference1->setText(QString::number(person.preference[1]));
	ui.lineEditPreference2->setText(QString::number(person.preference[2]));
	ui.lineEditPreference3->setText(QString::number(person.preference[3]));
	ui.lineEditPreference4->setText(QString::number(person.preference[4]));
	ui.lineEditPreference5->setText(QString::number(person.preference[5]));
	ui.lineEditPreference6->setText(QString::number(person.preference[6]));
	ui.lineEditPreference7->setText(QString::number(person.preference[7]));


	ui.lineEditFeature0->setText(QString::number(person.feature[0]));
	ui.lineEditFeature1->setText(QString::number(person.feature[1]));
	ui.lineEditFeature2->setText(QString::number(person.feature[2]));
	ui.lineEditFeature3->setText(QString::number(person.feature[3]));
	ui.lineEditFeature4->setText(QString::number(person.feature[4]));
	ui.lineEditFeature5->setText(QString::number(person.feature[5]));
	ui.lineEditFeature6->setText(QString::number(person.feature[6]));
	ui.lineEditFeature7->setText(QString::number(person.feature[7]));
}