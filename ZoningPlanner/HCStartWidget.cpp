#include "HCStartWidget.h"

HCStartWidget::HCStartWidget(QWidget* parent) : QDialog((QWidget*)parent) {
	// set up the UI
	ui.setupUi(this);

	ui.lineEditMaxRound->setText("3");
	ui.lineEditMaxStep->setText("3");

	connect(ui.okButton, SIGNAL(clicked()), this, SLOT(onOK()));
	connect(ui.cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
}

void HCStartWidget::onOK() {
	max_round = ui.lineEditMaxRound->text().toInt();
	max_step = ui.lineEditMaxStep->text().toInt();

	this->accept();
}
