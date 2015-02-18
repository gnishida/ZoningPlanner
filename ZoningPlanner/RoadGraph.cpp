﻿#include "RoadGraph.h"
#include <QGLWidget>
#include "GraphUtil.h"
#include "Util.h"
#include "global.h"

RoadGraph::RoadGraph() {
	modified = false;
}

RoadGraph::~RoadGraph() {
}

void RoadGraph::clear() {
	graph.clear();

	modified = true;
}
