#include "FLayer.h"
//#include "VBOUtil.h"

void FLayer::init(QVector3D _minPos, QVector3D _maxPos, int _imgResX, int _imgResY) {
	minPos=_minPos;
	maxPos=_maxPos;
	imgResX=_imgResX;
	imgResY=_imgResY;
		
	// Matrixを初期化（全要素を0.0で初期化）
	layerData = cv::Mat(imgResY, imgResX, CV_32F, cv::Scalar(0.0));

	// とりあえずテクスチャを生成
	updateTexFromData(0, 1);

	initialized=true;
}

void FLayer::updateTexFromData(float minValue, float maxValue) {
	if (texData != 0) {
		glDeleteTextures(1, &texData);
		texData = 0;
	}

	GLubyte* data = new GLubyte [layerData.cols * layerData.rows * 3];
	{
		int index = 0;
		for (int r = 0; r < layerData.rows; ++r) {
			for (int c = 0; c < layerData.cols; ++c) {
				int v = (layerData.at<float>(r, c) - minValue) / (maxValue - minValue) * 255;
				if (v > 255) v = 255;
				data[index * 3] = (GLubyte)v;
				data[index * 3 + 1] = (GLubyte)v;
				data[index * 3 + 2] = (GLubyte)v;
				index++;
			}
		}
	}

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);

	glGenTextures(1, &texData);
	glBindTexture(GL_TEXTURE_2D, texData);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, layerData.cols, layerData.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

	glGenerateMipmap(GL_TEXTURE_2D);

	delete [] data;
}

// return the interpolated value
float FLayer::getValue(float xM,float yM){
	int c1=floor(xM*imgResX);//0-imgRes
	int c2=c1+1;
	int r1=floor(yM*imgResY);
	int r2=r1+1;
	if (c1 < 0) c1 = 0;
	if (c1 >= layerData.cols) c1 = layerData.cols - 1;
	if (c2 < 0) c2 = 0;
	if (c2 >= layerData.cols) c2 = layerData.cols - 1;
	if (r1 < 0) r1 = 0;
	if (r1 >= layerData.rows) r1 = layerData.rows - 1;
	if (r2 < 0) r2 = 0;
	if (r2 >= layerData.rows) r2 = layerData.rows - 1;

	float v1 = layerData.at<uchar>(r1,c1);
	float v2 = layerData.at<uchar>(r2,c1);
	float v3 = layerData.at<uchar>(r1,c2);
	float v4 = layerData.at<uchar>(r2,c2);

	float v12,v34;
	if (yM*imgResY<=r1){
		v12 = v1;
		v34 = v3;
	} else if (yM*imgResY>=r2){
		v12 = v2;
		v34 = v4;
	} else {
		float s = yM*imgResY - r1;
		float t = r2 - yM*imgResY;
		v12 = (v1 * t + v2 * s) / (s + t);
		v34 = (v3 * t + v4 * s) / (s + t);
	}

	if (xM*imgResX<=c1){
		return v12;
	} else if (xM*imgResX>=c2){
		return v34;
	} else {
		float s = xM*imgResX - c1;
		float t = c2 - xM*imgResX;
		return (v12 * t + v34 * s) / (s + t);
	}
}

/*void FLayer::loadLayer(const QString& fileName) {
	layerData = cv::imread(fileName.toUtf8().data(), 0);//load one channel

	layerData /= 255.0f;

	// update image
	updateTexFromData(0, 1);
}
	
void FLayer::saveLayer(const QString& fileName) {
	cv::imwrite(fileName.toUtf8().data(), layerData);
}*/

