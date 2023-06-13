#ifndef MYRENDER_H
#define MYRENDER_H

#include <QLabel>
#include <QWidget>
#include <QMouseEvent>
#include <QWheelEvent>

class MyRender : public QLabel
{
    Q_OBJECT
public:
    MyRender(QWidget *parent = nullptr);
    int current_x,current_y,press_x,press_y;
    int x = 0,y = 0;
    int num_degrees = 0;

signals:
    void mouseMoved();
    void mousePressed();
    void wheelRolled();

protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
};

#endif // MYRENDER_H
