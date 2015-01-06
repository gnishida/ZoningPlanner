#include "MainWindow.h"
#include <QtGui/QApplication>
#include <numeric>
#include <algorithm>
#include "global.h"

int main(int argc, char *argv[])
{
	G::g["zoning_type_distribution"] = "0.5,0.2,0.1,0.1,0.05,0.05";
	G::g["zoning_start_size"] = 5;
	G::g["zoning_num_layers"] = 5;

	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
