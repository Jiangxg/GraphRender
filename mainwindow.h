#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "mycamera.h"
#include "PMPI/display.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    RenderLoader PMPISample = RenderLoader();
    std::string imagePath;
    int output_width;
    int output_height;
    unsigned char* out_data;
    int fps;

    MyCamera* camera = new MyCamera();
    void imgDisplayer(Ui::MainWindow* ui);
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

// signals:
//     void windowChanged();

// protected:
//     void changeEvent(QEvent *event);

private slots:
    void onMouseMoved();
    void updateProgressBar();
    void updateRender();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H


