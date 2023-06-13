#include "glm/glm/glm.hpp"
#include "glm/glm/gtc/matrix_transform.hpp"
#include <GL/gl.h> // for OpenGL 1.x
#include <GL/glu.h> // for OpenGL Utility Library (GLU)

#ifndef MYCAMERA_H
#define MYCAMERA_H
class MyCamera{
public:
    int screenWidth = 800;
    int screenHeight = 600;
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    glm::mat4 view;
    glm::mat3 R;
    glm::vec3 T;
    MyCamera();
    MyCamera(GLdouble posX, GLdouble posY, GLdouble posZ);
    void printCameraDetails() const;

};
// void controlCamera(MyCamera* camera, GLdouble posX, GLdouble posY, GLdouble posZ);
void controlCamera(MyCamera* camera, int current_x, int current_y, int num_degrees);
#endif // MYCAMERA_H
