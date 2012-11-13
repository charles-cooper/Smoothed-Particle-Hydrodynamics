#ifndef __API__
#define __API__



#if defined(_WIN32) || defined (_WIN64)
	#pragma comment( lib, "gl\\glut32.lib" )
	#include <windows.h>
	#include "..\gl\glut.h"
#else
	#include <string.h>
	#include <GL/glut.h>
#endif

//#include <gl\gl.h>
//#include <gl\glu.h>
//#include <gl\glaux.h>


#endif
