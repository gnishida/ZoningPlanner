#include "VBOLayer.h"
#include <QFileInfo>
#include "VBOUtil.h"
#include "qimage.h"
#include <QGLWidget>

#include "global.h"

#include "VBORenderManager.h"


VBOLayer::VBOLayer() {
	initialized=false;

	/// resolution of vertices
	resolutionX=200;
	resolutionY=200;
}



void VBOLayer::init(VBORenderManager& rendManager){

	QVector3D minPos=rendManager.minPos;
	QVector3D maxPos=rendManager.maxPos;
		

	//////////////////
	// TERRAIN LAYER
	if(initialized==false){
		//terrainLayer.init(minPos,maxPos,0,0,0,200,200);
		terrainLayer.init(minPos, maxPos, resolutionX, resolutionY, 0, resolutionX, resolutionY);

		glActiveTexture(GL_TEXTURE7);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glBindTexture(GL_TEXTURE_2D,terrainLayer.texData); 
		glActiveTexture(GL_TEXTURE0);
	}else{
		layer.minPos=minPos;
		layer.maxPos=maxPos;
	}

	//////////////////
	// VERTICES
	if(initialized==true){
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &elementbuffer);
		vbo=0;elementbuffer=0;

	}

	program=rendManager.program;

	std::vector<Vertex> vert;
	float sideX=abs(maxPos.x()-minPos.x())/resolutionX;
	float sideY=abs(maxPos.y()-minPos.y())/resolutionY;
		
	// VERTEX
	vert.push_back(Vertex(minPos, QVector3D(0, 0, 0)));
	vert.push_back(Vertex(QVector3D(maxPos.x(), minPos.y(), minPos.z()), QVector3D(1, 0, 0)));
	vert.push_back(Vertex(maxPos, QVector3D(1, 1, 0)));
	vert.push_back(Vertex(QVector3D(minPos.x(), maxPos.y(), minPos.z()), QVector3D(1, 0, 0)));

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*vert.size(), vert.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	initialized=true;
}//


void VBOLayer::render(VBORenderManager& rendManager ){//bool editionMode,QVector3D mousePos
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	GLuint vao;
	glGenVertexArrays(1,&vao); 
	glBindVertexArray(vao); 
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(

	glUniform1i (glGetUniformLocation (program, "mode"), 2);//MODE: terrain

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(4*sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(8*sizeof(float)));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(12*sizeof(float)));

	// Draw the triangles 
	glDrawElements(
		GL_QUADS, // mode
		indicesCount,    // count
		GL_UNSIGNED_SHORT,   // type
		(void*)0           // element array buffer offset
		);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
		glBindVertexArray(0);
	glDeleteVertexArrays(1,&vao);


}

void VBOLayer::updateTerrainNewValue(float coordX,float coordY,float newValue,float rad){
	layer.updateLayerNewValue(coordX,coordY,newValue,rad);

	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D,layer.texData); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glActiveTexture(GL_TEXTURE0);
}//

void VBOLayer::updateTerrain(float coordX,float coordY,float change,float rad){

	layer.updateLayer(coordX,coordY,change,rad);

	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D,layer.texData); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glActiveTexture(GL_TEXTURE0);
}//

float VBOLayer::getTerrainHeight(float xM,float yM){
	float value255=layer.getValue(xM,yM);
	const float maxHeight=7.0;//7=255*7 1785m (change in vertex as well)
	float height=maxHeight*value255;
	return height;
}//

void VBOLayer::loadTerrain(QString& fileName){
	layer.loadLayer(fileName);
	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D,layer.texData); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glActiveTexture(GL_TEXTURE0);
}//
	
void VBOLayer::saveTerrain(QString& fileName){
	layer.saveLayer(fileName);
}//

