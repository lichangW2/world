#include <iostream>
#include <vector>
#include <string.h>

using namespace std;


class Solution {
public:
    void  HeapSort(vector<int>& datas) {
        //刚开始就认为这个无序序列是一个堆(完全二叉树)
        // 从最后一个非叶子节点开始，调整使它成为一个大(小)顶堆
        int last_parent = (datas.size()-1) % 2 == 0 ? (datas.size()-1 - 2) / 2 : ((datas.size()-1) / 2); //从0开始
        int temp;
        for (; last_parent >= 0; last_parent--) {//从下向上，调整好大顶推
            adjust(datas,last_parent,datas.size());
        }

        for(int j=datas.size()-1;j>=0;j--){
            temp=datas[j];
            datas[j]=datas[0];
            datas[0]=temp;
            adjust(datas,0,j);
        }
    }

    void adjust (vector<int> & datas, int parent,int size){//每次都是从一个根向下调整，默认子树都已经调整好只有根父不平衡
        int temp;
        if ((2 * parent + 1) < size && \                 //要么影响左子树，要么影响右子树
                datas[2 * parent + 1] >= datas[parent] && \
                datas[2 * parent + 1] >= datas[2 * parent + 2]
                ) {
            temp = datas[2 * parent + 1];
            datas[2 * parent + 1]=datas[parent];
            datas[parent]=temp;
            adjust(datas,2 * parent+1,size);

        } else if (
                (2 * parent + 2) < size && \
                datas[2 * parent + 2] >= datas[parent] && \
                datas[2 * parent + 2] >= datas[2 * parent + 1]) {
            temp = datas[2 * parent + 2];
            datas[2 * parent + 2]=datas[parent];
            datas[parent]=temp;
            adjust(datas,2 * parent+2,size);

        }
        return;
    }


};


int main() {
    vector<int> pre{3, 4,65, 3,455 ,33, 4, 55, 32, 3, 21, 326, 44, 3, 6, 7, 7, 3, 5};

    Solution sloution;

     sloution.HeapSort(pre);

    for (int i = 0; i < pre.size(); ++i) {
        cout << pre[i] << endl;
    }
    return 0;

}