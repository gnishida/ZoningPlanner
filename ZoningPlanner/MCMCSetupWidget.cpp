#include "MCMCSetupWidget.h"

MCMCSetupWidget::MCMCSetupWidget(QWidget* parent) : QDialog((QWidget*)parent) {
	// set up the UI
	ui.setupUi(this);

	ui.lineEditInitialGridSize->setText("4");
	ui.lineEditNumStages->setText("5");
	ui.lineEditMCMCSteps->setText("200000");
	ui.lineEditUpscaleFactor->setText("1");

	connect(ui.okButton, SIGNAL(clicked()), this, SLOT(onOK()));
	connect(ui.cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
}

void MCMCSetupWidget::onOK() {
	initialGridSize = ui.lineEditInitialGridSize->text().toInt();
	numStages = ui.lineEditNumStages->text().toInt();
	MCMCSteps = ui.lineEditMCMCSteps->text().toInt();
	upscaleFactor = ui.lineEditUpscaleFactor->text().toFloat();

	this->accept();
}
