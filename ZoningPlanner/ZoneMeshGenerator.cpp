#include "ZoneMeshGenerator.h"

/**
 * Zoning の3Dモデルを生成する。
 */
void ZoneMeshGenerator::generateZoneMesh(VBORenderManager& rendManager, BlockSet& blocks) {
	rendManager.removeStaticGeometry("zoning");
	for (int i = 0; i < blocks.size(); ++i) {
		if (!blocks[i].valid) continue;
		if (blocks[i].zone.type() == ZoneType::TYPE_UNUSED) continue;

		for (int pi = 0; pi < blocks[i].parcels.size(); ++pi) {
			std::vector<Vertex> vert;

			QColor color;
			int opacity = 192;
			if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_RESIDENTIAL) {			// 住宅街（赤色）
				color = QColor(255 - (blocks[i].parcels[pi].zone.level() - 1) * 70, 0, 0, opacity);
			} else if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_COMMERCIAL) {	// 商業地（青色）
				color = QColor(0, 0, 255 - (blocks[i].parcels[pi].zone.level() - 1) * 70, opacity);
			} else if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_MANUFACTURING) {	// 工業地（灰色）
				int intensity = 200 - (blocks[i].parcels[pi].zone.level() - 1) * 60;
				color = QColor(intensity, intensity, intensity, opacity);
			} else if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_PARK) {			// 公園（緑色）
				color = QColor(0, 255, 0, opacity);
			} else if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_AMUSEMENT) {		// 繁華街（黄色）
				color = QColor(255, 255, 0, opacity);
			} else if (blocks[i].parcels[pi].zone.type() == ZoneType::TYPE_PUBLIC) {		// 公共施設（水色）
				color = QColor(0, 255, 255, opacity);
			} else {
				color = QColor(0, 0, 0, opacity);											// その他（黒色）
			}

			rendManager.addStaticGeometry2("zoning", blocks[i].parcels[pi].parcelContour.contour, 7.0f, false, "", GL_QUADS, 1|mode_AdaptTerrain, QVector3D(1, 1, 1), color);
		}
	}
}
