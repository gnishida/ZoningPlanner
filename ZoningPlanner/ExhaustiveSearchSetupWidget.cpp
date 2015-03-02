#include "ExhaustiveSearchSetupWidget.h"

ExhaustiveSearchSetupWidget::ExhaustiveSearchSetupWidget(QWidget* parent) : QDialog((QWidget*)parent) {
	// set up the UI
	ui.setupUi(this);

	ui.lineEditGridSize->setText("4");

	connect(ui.okButton, SIGNAL(clicked()), this, SLOT(onOK()));
	connect(ui.cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
}

void ExhaustiveSearchSetupWidget::onOK() {
	gridSize = ui.lineEditGridSize->text().toInt();

	this->accept();
}
