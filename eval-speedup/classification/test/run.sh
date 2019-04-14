export LD_LIBRARY_PATH=/opt/caffesp01/build/lib/:/opt/caffesp01/examples/cpp_classification:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
PATH=/usr/local/go/bin/:/opt/caffe//build/tools:/opt/caffe//python:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
rm libclassification.so ;rm main;rm main_queue; sh so2.sh; go build main_queue.go;./main_queue
