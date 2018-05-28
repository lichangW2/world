rm nohup.out
nohup docker build -t caffe2-cj:20180425 . &
tailf nohup.out
