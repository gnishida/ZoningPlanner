#include "MainWindow.h"
#include <QtGui/QApplication>
#include <numeric>
#include <algorithm>
#include "global.h"

int main(int argc, char *argv[])
{
	G::g["zoning_start_size"] = 5;
	G::g["zoning_num_layers"] = 5;

	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
