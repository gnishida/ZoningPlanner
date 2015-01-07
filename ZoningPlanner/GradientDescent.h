#pragma once

#include <vector>

class GradientDescent {
public:
	GradientDescent() {}

	void run(std::vector<float>& w, std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, int maxIterations, bool l1, float lambda, float eta, float threshold, bool normalize = true);

private:
	float negativeLogLikelihood(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, std::vector<float> w, bool l1, float lambda);
	float dot(std::vector<float> w, std::vector<float> f);
};

