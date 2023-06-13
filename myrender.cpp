#include "myrender.h"
#include <QDebug>

MyRender::MyRender(QWidget*parent) : QLabel(parent)
{

}

void MyRender::mouseMoveEvent(QMouseEvent *event)
{
    this->current_x = event->x();
    this->current_y = event->y();
    emit mouseMoved();
}

void MyRender::mousePressEvent(QMouseEvent *event)
{
    this->press_x = event->x();
    this->press_y = event->y();
    emit mousePressed();
}

void MyRender::wheelEvent(QWheelEvent *event)
{
    this->num_degrees += event->angleDelta().y();

    event->accept();
    emit wheelRolled();
}

