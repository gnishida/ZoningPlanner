#include "LPSolver.h"
#include <assert.h>

LPSolver::LPSolver(int num_variables) : num_variables(num_variables) {
	lp = make_lp(0, num_variables);

	// 変数名をx1, x2, ...のようにセットする
	for (int i = 0; i < num_variables; ++i) {
		char name[256];
		sprintf(name, "x%d", i + 1);

		// インデックスは１からスタートする！
		set_col_name(lp, i + 1, name);
	}

	adding_constraints = false;
}

LPSolver::~LPSolver() {
	if (lp) {
		delete_lp(lp);
	}
}

/**
 * 制約関数を追加する。
 * row、colnoには、non-zeroの係数だけセットすれば良い。
 * 例えば、1.2 * x1 + 4.5 * x3 = 2 なら、
 *     colno[0] = 1; row[0] = 1.2;
 *     colno[1] = 3; row[1] = 4.5;
 * となる。
 *
 * @param constr_type		式のタイプ (等号ならEQ、>=ならGE、<=ならLE）
 * @param row				各変数の係数のリスト
 * @param colno				変数のリスト
 * @param rh				右辺の値
 */
void LPSolver::addConstraint(int constr_type, std::vector<double>& row, std::vector<int>& colno, double rh) {
	if (!adding_constraints) {
		set_add_rowmode(lp, TRUE);
		adding_constraints = true;
	}

	add_constraintex(lp, row.size(), row.data(), colno.data(), constr_type, rh);
}

/**
 * Objective関数を定義する。
 * rowのサイズは、変数の数であること！
 *
 * @param row			各変数の係数
 */
void LPSolver::setObjective(std::vector<double>& row) {
	assert(row.size() == num_variables);

	objective_row.resize(row.size());
	std::copy(row.begin(), row.end(), objective_row.begin());

	if (adding_constraints) {
		set_add_rowmode(lp, FALSE);
	}

	// row配列のインデックス１からデータを格納する必要があるため、
	// 実際のサイズ＋１の配列を新たに確保し、コピーしている。
	std::vector<double> _row(row.size() + 1);
	std::copy(row.begin(), row.end(), _row.begin() + 1);

	set_obj_fn(lp, _row.data());
}

/**
 * 変数のupper boundをセットする。
 * valuesのサイズは、変数の数であること！
 *
 * @param values		各変数のupper boundのリスト
 */
void LPSolver::setUpperBound(std::vector<double>& values) {
	assert(values.size() == num_variables);

	for (int i = 0; i < values.size(); ++i) {
		// columnインデックスは１からスタートする！
		set_upbo(lp, i + 1, values[i]);
	}
}

int LPSolver::maximize() {
	set_maxim(lp);
	set_verbose(lp, IMPORTANT);
	return solve(lp);
}

int LPSolver::minimize() {
	set_minim(lp);
	set_verbose(lp, IMPORTANT);
	return solve(lp);
}

/**
 * Objective関数の値を返却する。
 *
 * @return			Objective関数の値
 */
double LPSolver::getObjective() {
	return get_objective(lp);
}

/**
 * index番目の変数の値を返却する。
 *
 * @param index		index番目の変数
 * @return			index番目の変数の値
 */
double LPSolver::getVariable(int index) {
	double* var;
	get_ptr_variables(lp, &var);

	return var[index + 1];
}

void LPSolver::displaySolution() {
	double obj = get_objective(lp);
	printf("Obj: %lf\n", obj);

	double* var;
	get_ptr_variables(lp, &var);
	for (int i = 0; i < num_variables; ++i) {
		if (var[i] == 0.0) continue;

		// column名は、インデックスが１からスタートする！
		printf("%s: %lf (%lf)\n", get_col_name(lp, i + 1), var[i], objective_row[i]);
	}
}
