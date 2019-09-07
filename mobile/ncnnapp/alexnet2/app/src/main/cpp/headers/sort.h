//
// Created by clausewang(王立昌) on 2019-06-01.
//

#ifndef ALEXNET_SORT_H
#define ALEXNET_SORT_H

#include <algorithm>
#include <vector>

template <typename T>
std::vector<std::pair<std::size_t,T> > sort_indexs(const std::vector<T>&v);
bool cmp(std::pair<std::size_t,float>& i1, std::pair<std::size_t,float>& i2);

//导致重复定义
//bool cmp(std::pair<size_t,float>& i1, std::pair<size_t,float>& i2){
//    return i1.second>i2.second;
//}


template <typename T>
std::vector<std::pair<std::size_t,T> > sort_indexs(const std::vector<T>&v){

    std::vector<std::pair<std::size_t,T> > nv(v.size());
    for (std::size_t i=0;i<v.size();i++){
        nv[i]=std::make_pair(i,v[i]);
    }
    std::sort(nv.begin(),nv.end(),cmp);

    return nv;
    //c++11 may not support
    // std::vector<size_t > idx(v.size());
    //    for(size_t i=0;i!=idx.size();i++){
    //        idx[i]=i;
    //    }
    //sort(idx.begin(),idx.end(),[&](size_t i1,size_t i2)->bool{return v[i1]<v[i2];});

}

#endif //ALEXNET_SORT_H
