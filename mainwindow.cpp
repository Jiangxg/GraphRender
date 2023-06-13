#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "myrender.h"
// #include "opencv2/opencv.hpp"
#include <iostream>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include "myresourcetracker.h"
#include "openglimagewidget.h"
#include <QSize>
#include <QEvent>
#include <QDebug>
#include <string>
#include "PMPI/display.h"
#include <cuda_runtime.h>
#include <chrono>
// #define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


void MainWindow::imgDisplayer(Ui::MainWindow* ui){
    // Convert the unsigned char* data to a QImage object
    QImage image(this->out_data, this->output_width,this->output_height,QImage::Format_RGB888);

    // Convert the QImage object to a QPixmap object
    QPixmap pixmap = QPixmap::fromImage(image);

    // Set the QPixmap object as the pixmap of the QLabel
    ui->Render->setPixmap(pixmap);
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->Render->setMouseTracking(false);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            qDebug() << "R[" << i << "][" << j << "] =" << this->camera->R[i][j];
        }
    }

    for (int i = 0; i < 3; i++) {
        qDebug() << "T[" << i << "] =" << this->camera->T[i];
    }

    // std::string filePath = "/home/jxg/jxg/GraphRender/img/img_lights.png";
    // Only for test
    std::string testPath = "/home/jxg/rendering/data/fern_192layers_c2f";
    this->imagePath = testPath;
    this->output_width = 1008;
    this->output_height = 756;

    // load the model files and allocate GPU memory
    readModel(this->imagePath, this->PMPISample, this->output_width, this->output_height);
    display(this->PMPISample, this->output_width, this->output_height, ui->Render->x, ui->Render->y);

    // 分配一块堆空间， 在程序生命周期内不变动
    this->out_data = new unsigned char[this->output_width * this->output_height * 3];

    //std::cout << "data_length = " << this->alpha_images.data.size() << "; height = " << this->alpha_images.height << "; width = " << this->alpha_images.width << "; channels = " << this->alpha_images.channels << std::endl;
    //std::cout << "data_length = " << this->images.data.size() << "; height = " << this->images.height << "; width = " << this->images.width << "; channels = " << this->images.channels << std::endl;

    //allocate_memory(this->alpha_images, this->images, this->dev_input_alpha, this->dev_input_k0, this->dev_output, 32);
    //display(this->dev_input_alpha, this->dev_input_k0, this->dev_output, this->images.width, this->images.height, this->images.channels, 32, ui->Render->x, ui->Render->y);
    //retrieveMemory(this->in_out, this->dev_output, this->images.width, this->images.height, this->images.channels);
    // display(this->alpha_images, this->images, this->in_out, 32);

    this->imgDisplayer(ui);


    QTimer* ProgressDisplayTimer = new QTimer(this);
    QTimer* UpdateProgressTimer = new QTimer(this);
    QTimer* RenderUpdate = new QTimer(this);

    connect(ui->Render,SIGNAL(mouseMoved()),this,SLOT(onMouseMoved()));
    connect(ui->Render,&MyRender::wheelRolled,this,&MainWindow::onMouseMoved);
    connect(ProgressDisplayTimer, &QTimer::timeout, this, &MainWindow::updateProgressBar);
    connect(UpdateProgressTimer, &QTimer::timeout, ui->GPUTrack, &MyResourceTracker::mockingValueUpdate);
    connect(RenderUpdate, &QTimer::timeout, this, &MainWindow::updateRender);
    
    ProgressDisplayTimer->start(10);
    UpdateProgressTimer->start(100);
    RenderUpdate->start(10);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onMouseMoved()
{
    ui->Render->x += (ui->Render->current_x - ui->Render->press_x);
    ui->Render->y += (ui->Render->current_y - ui->Render->press_y);
    // auto start = std::chrono::high_resolution_clock::now();
    controlCamera(this->camera, ui->Render->x, ui->Render->y, ui->Render->num_degrees);
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    // this->camera->printCameraDetails();
}

void MainWindow::updateProgressBar(){
    ui->GPUTrack->setValue(ui->GPUTrack->resourceConsumingRate);
    ui->CPUTrack->setValue(ui->CPUTrack->resourceConsumingRate);
    ui->GPURate->setText(QString("%1 %2").arg(this->fps).arg(this->output_width));
    ui->CPURate->setText(QString("%1 %2").arg(ui->Render->y).arg(this->output_height));
}

void MainWindow::updateRender(){
    // auto start = std::chrono::high_resolution_clock::now();

    // TODO: 传入camera参数
    //std::cout << "running updateRender" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    display(this->PMPISample, this->output_width, this->output_height, ui->Render->x, ui->Render->y);
    
    retrieveMemory(this->out_data, this->PMPISample.dev_output, this->output_width, this->output_height);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Rendering time: " << duration.count() << " microseconds" << std::endl;
    this->fps = 1000000 / duration.count();
    this->imgDisplayer(ui);
}