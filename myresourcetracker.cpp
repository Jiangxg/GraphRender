#include "myresourcetracker.h"

MyResourceTracker::MyResourceTracker(QWidget *parent) : QProgressBar(parent)
{

}

void MyResourceTracker::valueUpdate(int updateValue)
{
    this->resourceConsumingRate = updateValue;
}

void MyResourceTracker::mockingValueUpdate()
{
    if(this->resourceConsumingRate == 100)
        this->resourceConsumingRate = 0;
    else
        this->resourceConsumingRate += 1;
}
