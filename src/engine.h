#include "api.h"

#ifndef __ENGINE
#define __ENGINE

class Engine {
private:
	GLsizei Height, Width;
	void SetProjectionMatrix(void);
	void SetModelviewMatrix(void);
public:
	void Resize(GLsizei width, GLsizei height);
	void Init(void);
	void Draw(void);
};

#endif
