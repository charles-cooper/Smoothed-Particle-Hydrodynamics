#include "engine.h"

GLvoid Engine::SetProjectionMatrix(GLvoid){
	glMatrixMode(GL_PROJECTION);						// Действия будут производиться с матрицей проекции
	glLoadIdentity();									// Текущая матрица (проекции) сбрасывается на единичную
	glFrustum(-1, 1, -1, 1, 3, 10);						// Устанавливается перспективная проекция
}

GLvoid Engine::SetModelviewMatrix(GLvoid){
     glMatrixMode(GL_MODELVIEW);                                   // Äåéñòâèÿ áóäóò ïðîèçâîäèòüñÿ ñ ìàòðèöåé ìîäåëè
     glLoadIdentity();                                             // Òåêóùàÿ ìàòðèöà (ìîäåëè) ñáðàñûâàåòñÿ íà åäèíè÷íóþ
     glTranslatef(0.0, 0.0, -8.0);                              // Ñèñòåìà êîîðäèíàò ïåðåíîñèòñÿ íà 8 åäèíèö âãëóáü ñöåíû                                                                 
     //glRotatef(30.0, 1.0, 0.0, 0.0);                              // è ïîâîðà÷èâàåòñÿ íà 30 ãðàäóñîâ âîêðóã îñè x,
     glOrtho(-0.7, 0.7, -0.3, 0.7, 0, 15);
     //glRotatef(90.0, 0.0, 0.0, 1.0);                              // à çàòåì íà 70 ãðàäóñîâ âîêðóã îñè y
     //glRotatef(90.0, 1.0, 0.0, 0.0);                              // à çàòåì íà 70 ãðàäóñîâ âîêðóã îñè y
}


GLvoid Engine::Resize(GLsizei width, GLsizei height){
	if (height == 0)									
	{
		height = 1;										
	}

	glViewport(0, 0, width, height);					// Устанавливается область просмотра

	Height = height;
	Width = width;

	SetProjectionMatrix();
	SetModelviewMatrix();
}

GLvoid Engine::Init(GLvoid){
	glClearColor(0.2f, 0.5f, 0.75f, 1.0f);				// Устанавливается синий фон
	glClearDepth(1.0f);									// Устанавливается значение для
														// заполнения буфера глубины по умолчанию
	glEnable(GL_DEPTH_TEST);							// Включается тест глубины
	glDepthFunc(GL_LEQUAL);								// Устанавливается значение, используемое
														// в сравнениях при использовании
														// буфера глубины

	glShadeModel(GL_SMOOTH);							// Включается плавное затенение
	glEnable(GL_LINE_SMOOTH);							// Включается сглаживание линий
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);				// Выбирается самый качественный
														// режим сглаживания для линий
	glEnable(GL_BLEND);									// Включается смешение цветов, необходимое
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// для работы сглаживания и задается
														// способ смешения
}

GLvoid Engine::Draw(GLvoid)									
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Очищается буфер кадра и буфер глубины
	glPushMatrix();										// Запоминается матрица модели

	glColor3f(1.0f, 1.0f, 1.0f);						// Задается текущий цвет примитивов
	glutWireCube(2.0f);									// Рисуется проволочный куб со стороной 2

	glPopMatrix();										// Восстанавливается матрица модели
}
