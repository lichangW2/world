g++ main.cpp -DUSE_OPENCV  -I/opt/caffesp01/include -I/usr/local/cuda-8.0/targets/x86_64-linux/include/ -I/opt/caffesp01/.build_release/src  -L/usr/lib/x86_64-linux-gnu/ -L/opt/caffesp01/.build_release/lib/ -L./ -lclassification -lboost_system -lpthread -lgflags -lunwind -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core  -lglog -lcaffe  -std=c++11
