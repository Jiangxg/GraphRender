QT       += core gui
QT += opengl
#QT += openglwidgets


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += /usr/local/cuda-11.1/include
INCLUDEPATH += /glm
HEADERS += stb/stb_image.h
LIBS += -L -ldisplaydemo
LIBS += -L/usr/local/cuda-11.1/lib64 -lcudart
LIBS += -lGLEW

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    myrender.cpp \
    myresourcetracker.cpp \
    openglimagewidget.cpp \
    mycamera.cpp \
    PMPI/display.cu \
    PMPI/aabb/src/intersect_gpu.cu

HEADERS += \
    mainwindow.h \
    myrender.h \
    myresourcetracker.h \
    openglimagewidget.h \
    mycamera.h \ 
    PMPI/display.h 

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
