package main

// #cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu/ -L/opt/caffesp01/.build_release/lib/ -L./ -lclassification -lboost_system -lpthread -lgflags -lunwind -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core  -lglog -lcaffe
// #cgo CXXFLAGS: -I/opt/caffesp01/include -I/usr/local/cuda-8.0/targets/x86_64-linux/include/ -I/opt/caffesp01/.build_release/src  -std=c++11
// #include <stdlib.h>
// #include "classification.h"
import "C"
import "unsafe"

import (
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"sync"
	"encoding/json"
	"context"
	"time"
	"fmt"
)

var fine_ctx *C.classifier_ctx
var coarse_ctx *C.classifier_ctx
var det_ctx *C.classifier_ctx

type SingleRequest struct {
	data    *byte
	length  int 
	ch	chan string
}

var req_channel = make(chan SingleRequest, 1000) //max queue length

func startHandleRoutines(routine_num int){
	fmt.Printf("startng %d handle routines ", routine_num)
	for i:=0; i < routine_num; i++ {
		go func(){
			mux := sync.Mutex{}
			for {
				var s SingleRequest  
				select {
					case  s = <- req_channel:
				}
				fmt.Printf("%v\n", s)


				results := make(map[string]string)
				wg := sync.WaitGroup{}

        wg.Add(1)
        go func(){
                defer wg.Done()
                cstr, err := C.classifier_classify(fine_ctx, (*C.char)(unsafe.Pointer(s.data)), C.size_t(s.length))
                if err != nil {
                        //http.Error(w, err.Error(), http.StatusBadRequest)
                        return
                }
                defer C.free(unsafe.Pointer(cstr))
                mux.Lock()
                defer mux.Unlock()
                results["fine"] = C.GoString(cstr)
        }()


        wg.Add(1)
        go func(){
                defer wg.Done()
                cstr, err := C.classifier_classify(coarse_ctx, (*C.char)(unsafe.Pointer(s.data)), C.size_t(s.length))
                if err != nil {
                        //http.Error(w, err.Error(), http.StatusBadRequest)
                        return
                }
                defer C.free(unsafe.Pointer(cstr))
                mux.Lock()
                defer mux.Unlock()
                results["coarse"] = C.GoString(cstr)
        }()


        wg.Add(1)
        go func(){
                defer wg.Done()
                cstr, err := C.classifier_classify(det_ctx, (*C.char)(unsafe.Pointer(s.data)), C.size_t(s.length))
                if err != nil {
                        //http.Error(w, err.Error(), http.StatusBadRequest)
                        return
                }
                defer C.free(unsafe.Pointer(cstr))
                mux.Lock()
                defer mux.Unlock()
                results["det"] = C.GoString(cstr)
        }()


        wg.Wait()

        jsonStr, err := json.Marshal(results)
	if err != nil {
		fmt.Printf("json error!\n");
                return
        }
	fmt.Printf("%s\n", string(jsonStr))
	s.ch <- string(jsonStr)

			}
		}()
	}
}

func classify(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	buffer, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var tctx, cancelFunc = context.WithTimeout(context.Background(), 10*time.Second)

	defer func() {
		cancelFunc()
	}()

	fmt.Printf("%p %p %p ", buffer, &buffer,  &buffer[0])
	req := SingleRequest{
                        data : &buffer[0],
                        length : len(buffer),
                        ch : make(chan string),
        }

	go func(ctx context.Context){
		req_channel <- req
	}(tctx)

	var results string
	select {
		case results = <- req.ch:
			fmt.Printf("got result --------:%s\n", results)
		case <- tctx.Done():
			fmt.Printf("Timeout")
			http.Error(w, "Timeout", http.StatusBadRequest)
	}
/*
	jsonStr, err := json.Marshal(results)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
*/
	io.WriteString(w, results)
}

func main() {
	var err error
	log.Println("Initializing Caffe classifiers")


	fine_model := C.CString("/workspace/serving/models/online-model/used/fine_deploy.prototxt")
	fine_trained := C.CString("/workspace/serving/models/online-model/used/fine_weight.caffemodel")
	fine_name := C.CString("fine")
	fine_ctx, err = C.classifier_initialize(fine_name, fine_model, fine_trained, 5)
	if err != nil {
		log.Fatalln("could not initialize classifier:", err)
		return
	}
	defer C.classifier_destroy(fine_ctx)

	coarse_model := C.CString("/workspace/serving/models/online-model/used/coarse_deploy.prototxt")
	coarse_trained := C.CString("/workspace/serving/models/online-model/used/coarse_weight.caffemodel")
	coarse_name := C.CString("coarse")
	coarse_ctx, err = C.classifier_initialize(coarse_name, coarse_model, coarse_trained, 5)
        if err != nil {
                log.Fatalln("could not initialize classifier:", err)
                return
        }
        defer C.classifier_destroy(coarse_ctx)

        det_model := C.CString("/workspace/serving/models/online-model/used/det_deploy.prototxt")
        det_trained := C.CString("/workspace/serving/models/online-model/used/det_weight.caffemodel")
	det_name := C.CString("det")
        det_ctx, err = C.classifier_initialize(det_name, det_model, det_trained, 5)
        if err != nil {
                log.Fatalln("could not initialize classifier:", err)
                return
        }
        defer C.classifier_destroy(det_ctx)


	startHandleRoutines(10);	

	log.Println("Adding REST endpoint /api/classify")
	http.HandleFunc("/api/classify", classify)
	log.Println("Starting server listening on :8000")
	log.Fatal(http.ListenAndServe(":8000", nil))
}

