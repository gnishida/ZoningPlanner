#pragma once

#include <lp_lib.h>
#include <vector>

class LPSolver {
private:
	lprec *lp;
	int num_variables;
	bool adding_constraints;
	std::vector<double> objective_row;

public:
	LPSolver(int num_variables);
	~LPSolver();

	void addConstraint(int constr_type, std::vector<double>& row, std::vector<int>& colno, double rh);
	void setObjective(std::vector<double>& row);
	void setUpperBound(std::vector<double>& values);
	int maximize();
	int minimize();
	double getObjective();
	double getVariable(int index);
	void displaySolution();
};

