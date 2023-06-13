#ifndef OPENGLIMAGEWIDGET_H
#define OPENGLIMAGEWIDGET_H
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

class OpenGLImageWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    OpenGLImageWidget(QWidget *parent = nullptr);

    void setImage(unsigned char *imageData, int width, int height);

protected:
    void initializeGL() override;
    void paintGL() override;

private:
    int imageWidth;
    int imageHeight;
    unsigned char *image;
};

#endif // OPENGLIMAGEWIDGET_H
