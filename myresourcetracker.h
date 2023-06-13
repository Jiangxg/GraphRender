#ifndef MYRESOURCETRACKER_H
#define MYRESOURCETRACKER_H

#include <QProgressBar>
#include <QWidget>

class MyResourceTracker : public QProgressBar
{
    Q_OBJECT
public:
    MyResourceTracker(QWidget *parent = nullptr);

    int resourceConsumingRate = 0;

    void valueUpdate(int updateValue);
    void mockingValueUpdate();
};

#endif // MYRESOURCETRACKER_H
