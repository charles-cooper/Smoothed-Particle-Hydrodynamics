#include "engine.h"

extern float rotX;

void Engine::SetProjectionMatrix(void){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();									// Current projection matrix is dropped to identity matrix 
	glFrustum(-1, 1, -1, 1, 3, 15);						// Set up perspective projection
}

void Engine::SetModelviewMatrix(void){
     glMatrixMode(GL_MODELVIEW);                                   
     glLoadIdentity();                                             
     glTranslatef(0.0, 0.0, -8.0);                              
     glRotatef(10.0, 1.0, 0.0, 0.0);
   //  glOrtho(-0.7*2, 0.7*2, -0.3*2, 0.7*2, 0, 15);
     //glRotatef(90.0, 0.0, 0.0, 1.0);
     glRotatef(rotX, 0.0, 1.0, 0.0);                              
}

GLvoid Engine::Resize(GLsizei width, GLsizei height){
	if(height == 0)
	{
		height = 1;										
	}

	glViewport(0, 0, width, height);					// Sev view area

	Height = height;
	Width = width;

	SetProjectionMatrix();
	SetModelviewMatrix();
}

void Engine::Init(void){
	glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

}

void Engine::Draw(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();

	glColor3f(1.0f, 1.0f, 1.0f);
	//glutWireCube(2.0f);

	glPopMatrix();
}
