#include "MainWindow.h"
#include <QtGui/QApplication>
#include <numeric>
#include <algorithm>
#include "global.h"

int main(int argc, char *argv[])
{
	G::g["tree_setback"] = 1.0f;
	G::g["zoning_type_distribution"] = "0.65,0.1,0.05,0.1,0.05,0.05";
	G::g["zoning_start_size"] = 5;
	G::g["zoning_num_layers"] = 5;
	G::g["preference_for_land_value"] = "0.534, -0.01, 0.267, 0.356, 0.089, 0.0, -0.713";

	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
