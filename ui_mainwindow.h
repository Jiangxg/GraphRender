/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <myrender.h>
#include <myresourcetracker.h>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    MyRender *Render;
    QFrame *MonitorArea;
    QGridLayout *gridLayout_3;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_3;
    QLabel *label;
    QLabel *label_2;
    QVBoxLayout *verticalLayout;
    MyResourceTracker *GPUTrack;
    MyResourceTracker *CPUTrack;
    QVBoxLayout *verticalLayout_2;
    QLabel *GPURate;
    QLabel *CPURate;
    QFrame *ControlArea;
    QLabel *AlgotithmOption;
    QListWidget *listWidget;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(806, 617);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        frame = new QFrame(centralwidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setStyleSheet(QStringLiteral("background-color: rgb(33, 33, 33)"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        Render = new MyRender(frame);
        Render->setObjectName(QStringLiteral("Render"));
        Render->setMinimumSize(QSize(640, 360));
        Render->setMouseTracking(true);
        Render->setStyleSheet(QLatin1String("color: rgb(255, 255, 255);\n"
"background-color: rgb(94, 94, 94)"));
        Render->setFrameShape(QFrame::NoFrame);
        Render->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(Render, 0, 1, 1, 1);

        MonitorArea = new QFrame(frame);
        MonitorArea->setObjectName(QStringLiteral("MonitorArea"));
        MonitorArea->setMinimumSize(QSize(640, 140));
        MonitorArea->setMaximumSize(QSize(16777215, 140));
        MonitorArea->setStyleSheet(QStringLiteral("background-color:rgb(94, 94, 94)"));
        MonitorArea->setFrameShape(QFrame::StyledPanel);
        MonitorArea->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(MonitorArea);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        label = new QLabel(MonitorArea);
        label->setObjectName(QStringLiteral("label"));
        label->setStyleSheet(QStringLiteral("color:rgb(255, 255, 255)"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        verticalLayout_3->addWidget(label);

        label_2 = new QLabel(MonitorArea);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setStyleSheet(QStringLiteral("color:rgb(255, 255, 255)"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        verticalLayout_3->addWidget(label_2);


        horizontalLayout->addLayout(verticalLayout_3);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        GPUTrack = new MyResourceTracker(MonitorArea);
        GPUTrack->setObjectName(QStringLiteral("GPUTrack"));
        GPUTrack->setStyleSheet(QLatin1String("QProgressBar {\n"
"  border: 2px ;\n"
"  border-radius: 2px;\n"
"  background-color: rgb(214, 214, 214);\n"
"  text-align: center;\n"
"color: transparent;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"  background-color: rgb(0, 150, 255);\n"
"  border-radius: 2px;\n"
"}\n"
""));
        GPUTrack->setValue(24);

        verticalLayout->addWidget(GPUTrack);

        CPUTrack = new MyResourceTracker(MonitorArea);
        CPUTrack->setObjectName(QStringLiteral("CPUTrack"));
        CPUTrack->setStyleSheet(QLatin1String("QProgressBar {\n"
"  border: 2px ;\n"
"  border-radius: 2px;\n"
"  background-color: rgb(214, 214, 214);\n"
"  text-align: center;\n"
"color: transparent;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"  background-color: rgb(0, 150, 255);\n"
"  border-radius: 2px;\n"
"}\n"
""));
        CPUTrack->setValue(24);

        verticalLayout->addWidget(CPUTrack);


        horizontalLayout->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        GPURate = new QLabel(MonitorArea);
        GPURate->setObjectName(QStringLiteral("GPURate"));
        GPURate->setStyleSheet(QStringLiteral("color:rgb(255,255,255)"));

        verticalLayout_2->addWidget(GPURate);

        CPURate = new QLabel(MonitorArea);
        CPURate->setObjectName(QStringLiteral("CPURate"));
        CPURate->setStyleSheet(QStringLiteral("color:rgb(255,255,255)"));

        verticalLayout_2->addWidget(CPURate);


        horizontalLayout->addLayout(verticalLayout_2);


        gridLayout_3->addLayout(horizontalLayout, 0, 0, 1, 1);


        gridLayout_2->addWidget(MonitorArea, 1, 1, 1, 1);

        ControlArea = new QFrame(frame);
        ControlArea->setObjectName(QStringLiteral("ControlArea"));
        ControlArea->setMinimumSize(QSize(100, 500));
        ControlArea->setMaximumSize(QSize(100, 16777215));
        ControlArea->setStyleSheet(QStringLiteral("background-color:rgb(94, 94, 94)"));
        ControlArea->setFrameShape(QFrame::StyledPanel);
        ControlArea->setFrameShadow(QFrame::Raised);
        AlgotithmOption = new QLabel(ControlArea);
        AlgotithmOption->setObjectName(QStringLiteral("AlgotithmOption"));
        AlgotithmOption->setGeometry(QRect(13, 13, 62, 16));
        AlgotithmOption->setStyleSheet(QStringLiteral("color: rgb(255, 255, 255)"));
        listWidget = new QListWidget(ControlArea);
        new QListWidgetItem(listWidget);
        new QListWidgetItem(listWidget);
        new QListWidgetItem(listWidget);
        listWidget->setObjectName(QStringLiteral("listWidget"));
        listWidget->setGeometry(QRect(13, 37, 74, 91));
        listWidget->setStyleSheet(QLatin1String("QListWidget {\n"
"    background-color: rgb(94, 94, 94);\n"
"    font-family: Arial;\n"
"    font-size: 12pt;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    height: 30px;\n"
"    background-color: transparent;\n"
"    border: 1px;\n"
"    color: white;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: rgb(33, 33, 33);\n"
"}\n"
"\n"
"QListWidget::item:hover {\n"
"    background-color: rgb(66, 66, 66);\n"
"}\n"
"\n"
"QListWidget::item:selected:active {\n"
"    background-color: rgb(33, 33, 33);\n"
"    color: white;\n"
"}\n"
"\n"
"\n"
""));
        listWidget->setFrameShape(QFrame::NoFrame);
        listWidget->setFrameShadow(QFrame::Plain);
        listWidget->setLineWidth(0);
        // listWidget->setItemAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(ControlArea, 0, 0, 2, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 806, 22));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "GraphRender", Q_NULLPTR));
        Render->setText(QString());
        label->setText(QApplication::translate("MainWindow", "GPU:", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "CPU:", Q_NULLPTR));
        GPURate->setText(QApplication::translate("MainWindow", "0%", Q_NULLPTR));
        CPURate->setText(QApplication::translate("MainWindow", "0%", Q_NULLPTR));
        AlgotithmOption->setText(QApplication::translate("MainWindow", "Mode:", Q_NULLPTR));

        const bool __sortingEnabled = listWidget->isSortingEnabled();
        listWidget->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem = listWidget->item(0);
        ___qlistwidgetitem->setText(QApplication::translate("MainWindow", "mode 1", Q_NULLPTR));
        QListWidgetItem *___qlistwidgetitem1 = listWidget->item(1);
        ___qlistwidgetitem1->setText(QApplication::translate("MainWindow", "mode 2", Q_NULLPTR));
        QListWidgetItem *___qlistwidgetitem2 = listWidget->item(2);
        ___qlistwidgetitem2->setText(QApplication::translate("MainWindow", "mode 3", Q_NULLPTR));
        listWidget->setSortingEnabled(__sortingEnabled);

    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
