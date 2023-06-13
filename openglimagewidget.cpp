#include "openglimagewidget.h"

OpenGLImageWidget::OpenGLImageWidget(QWidget *parent) : QOpenGLWidget(parent), imageWidth(0), imageHeight(0), image(nullptr)
{

}

void OpenGLImageWidget::setImage(unsigned char *imageData, int width, int height)
{
    image = imageData;
    imageWidth = width;
    imageHeight = height;
    update();
}

void OpenGLImageWidget::initializeGL()
{
    initializeOpenGLFunctions();

}

void OpenGLImageWidget::paintGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (image) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glDrawPixels(imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, image);
    }
}

