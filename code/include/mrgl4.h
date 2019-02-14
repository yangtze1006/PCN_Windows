#pragma once

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "gl/glew.h"
#include "glm/glm.hpp"
#include "stdio.h"

#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"glfw3.lib")

void glinfo()
{
	const GLubyte* name = glGetString(GL_VENDOR); //���ظ���ǰOpenGLʵ�ֳ��̵�����  
	const GLubyte* biaoshifu = glGetString(GL_RENDERER); //����һ����Ⱦ����ʶ����ͨ���Ǹ�Ӳ��ƽ̨  
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION); //���ص�ǰOpenGLʵ�ֵİ汾��  
	const GLubyte* gluVersion = gluGetString(GLU_VERSION); //���ص�ǰGLU���߿�汾  
	printf("OpenGLʵ�ֳ��̵����֣�%s\n", name);
	printf("��Ⱦ����ʶ����%s\n", biaoshifu);
	printf("OOpenGLʵ�ֵİ汾�ţ�%s\n", OpenGLVersion);
	printf("OGLU���߿�汾��%s\n", gluVersion);
}