cmake_minimum_required(VERSION 3.26)
project(opencv-projects)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

set(CMAKE_CXX_STANDARD 17)

function(add_chapter TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME}
            PUBLIC
            ${CMAKE_CURRENT_LIST_DIR}
            ${OPENCV_INCLUDE_DIRS}
    )
    target_link_libraries(${TARGET_NAME}
            PUBLIC
            ${OPENCV_LIBRARIES}
    )
endfunction()

# Virtual Painter
add_chapter(virtual-painter VirtualPainter/main.cpp)

# License Plate Detector
add_chapter(license-plate-detector LicensePlateDetector/main.cpp)

# Face Tracking
add_chapter(face-tracking FaceTracking/main.cpp)

# Soccer Field Segmentation
add_chapter(color-segmentation ColorSegmentation/main.cpp)

# Document Scanner
add_chapter(document-scanner DocumentScanner/main.cpp)

# Basic Object Detection
add_chapter(shape-detection BasicShapeDetection/main.cpp)

# Image Warp
add_chapter(image-warping ImageWarping/main.cpp)

# Camera Calibration
add_chapter(camera-calibration CameraCalibration/main.cpp)

# Fourier Transform
add_chapter(dfourier-transform DiscreteFourierTransform/main.cpp)
add_chapter(dfourier-transform-implementation DiscreteFourierTransform/dftImplementation.cpp)

# Histograms
add_chapter(histograms Histograms/main.cpp)

# Corner Detection
add_chapter(shi-tomasi-corner-detection CornerDetection/shiTomasiCornerDetection.cpp)
add_chapter(harris-corner-detection CornerDetection/harrisCornerDetection.cpp)

# Color Picker
add_chapter(color-picker ColorDetection/main.cpp)

# Template Matching
add_chapter(template-matching TemplateMatching/findMatch.cpp)
add_chapter(template-matching-all TemplateMatching/findAllMatches.cpp)