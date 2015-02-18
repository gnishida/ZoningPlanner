#include "ZoneMeshGenerator.h"

/**
 * Zoning の3Dモデルを生成する。
 */
void ZoneMeshGenerator::generateZoneMesh(VBORenderManager& rendManager, BlockSet& blocks) {
	rendManager.removeStaticGeometry("zoning");
	for (int i = 0; i < blocks.size(); ++i) {
		if (!blocks[i].valid) continue;
		
		if (blocks[i].zone.type() == ZoneType::TYPE_UNUSED) continue;

		// Blockの3Dモデルを生成（Block表示モードの時にのみ、表示される）
		{
			std::vector<Vertex> vert;

			QColor color;
			int opacity = 192;
			if (i == blocks.selectedBlockIndex) {
				color = QColor(255, 255, 255, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_RESIDENTIAL) {	// 住宅街は赤色ベース
				color = QColor(255 - (blocks[i].zone.level() - 1) * 70, 0, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_COMMERCIAL) {	// 商業地は青色ベース
				color = QColor(0, 0, 255 - (blocks[i].zone.level() - 1) * 70, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_MANUFACTURING) {	// 工場街は灰色ベース
				color = QColor(200 - (blocks[i].zone.level() - 1) * 60, 150 - (blocks[i].zone.level() - 1) * 40, 200 - (blocks[i].zone.level() - 1) * 60, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_PARK) {			// 公園は緑色
				color = QColor(0, 255, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_AMUSEMENT) {		// 繁華街は黄色
				color = QColor(255, 255, 0, opacity);
			} else if (blocks[i].zone.type() == ZoneType::TYPE_PUBLIC) {		// 公共施設は水色ベース
				color = QColor(0, 255, 255, opacity);
			} else {
				color = QColor(128, 128, 128, opacity);
			}

			rendManager.addStaticGeometry2("zoning", blocks[i].blockContour.contour, 8.0f, false, "", GL_QUADS, 1|mode_AdaptTerrain, QVector3D(1, 1, 1), color);
		}
	}
}
