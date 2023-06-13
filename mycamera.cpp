#include <iostream>
#include "mycamera.h"

MyCamera::MyCamera(){
    this->cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    this->cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    this->cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    this->view = glm::lookAt(this->cameraPos, this->cameraPos + this->cameraFront, this->cameraUp);
    this->R = glm::mat3(this->view);
    this->T = glm::vec3(this->view[3]);
}

MyCamera::MyCamera(GLdouble posX, GLdouble posY, GLdouble posZ){
    this->cameraPos = glm::vec3(posX, posY, posZ);
    this->cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    this->cameraUp = glm::vec3(posX, posY, posZ - 1.0f);

    this->view = glm::lookAt(this->cameraPos, this->cameraPos + this->cameraFront, this->cameraUp);
    this->R = glm::mat3(this->view);
    this->T = glm::vec3(this->view[3]);
}

void MyCamera::printCameraDetails() const {
    std::cout << "Camera Details:" << std::endl;
    std::cout << "Position: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
    std::cout << "Front Direction: (" << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z << ")" << std::endl;
    std::cout << "Up Direction: (" << cameraUp.x << ", " << cameraUp.y << ", " << cameraUp.z << ")" << std::endl;
}

// void controlCamera(MyCamera* camera, GLdouble posX, GLdouble posY, GLdouble posZ){
//     camera->cameraPos = glm::vec3(posX, posY, posZ);
//     camera->cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
//     camera->cameraUp = glm::vec3(posX, posY, posZ - 1.0f);

//     camera->view = glm::lookAt(camera->cameraPos, camera->cameraPos + camera->cameraFront, camera->cameraUp);
//     camera->R = glm::mat3(camera->view);
//     camera->T = glm::vec3(camera->view[3]);
// }
void controlCamera(MyCamera* camera, int current_x, int current_y, int num_degrees) {
    // Normalize the mouse coordinates to a range of -1 to 1
    GLdouble norm_x = static_cast<GLdouble>(current_x) / camera->screenWidth * 2 - 1;
    GLdouble norm_y = 1 - static_cast<GLdouble>(current_y) / camera->screenHeight * 2;

    // Convert the normalized coordinates to angles (in radians)
    GLdouble horizontalAngle = norm_x * M_PI;
    GLdouble verticalAngle = norm_y * M_PI;

    // Convert the number of degrees to a zoom factor
    GLdouble zoomFactor = std::pow(1.0015, num_degrees);

    // Calculate the camera position using the angles and zoom factor
    GLdouble posX = zoomFactor * std::cos(verticalAngle) * std::sin(horizontalAngle);
    GLdouble posY = zoomFactor * std::cos(verticalAngle) * std::cos(horizontalAngle);
    GLdouble posZ = zoomFactor * std::sin(verticalAngle);

    // Update the camera position, front, and up vectors
    camera->cameraPos = glm::vec3(posX, posY, posZ);
    camera->cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    camera->cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    // Update the view matrix, rotation matrix, and translation vector
    camera->view = glm::lookAt(camera->cameraPos, camera->cameraPos + camera->cameraFront, camera->cameraUp);
    camera->R = glm::mat3(camera->view);
    camera->T = glm::vec3(camera->view[3]);
}
