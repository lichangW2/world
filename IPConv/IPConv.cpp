//
// Created by CJ on 11/04/2019.
//

#include <iostream>
#include "IPConv.h"

using namespace std;

#define MIN_LEN 7

int IPConv(const char *ipstr, unsigned int* result){

    unsigned int len=strlen(ipstr);

    //====================================
    if (ipstr== nullptr||len<MIN_LEN) {
        cerr << "invalid ip address length" << endl;
        return -1;
    }
    if (result== nullptr){
        cerr<<"invalid result address"<<endl;
        return -1;
    }

    //====================================

    *result=0;
    int counter=0;
    char last='a';
    int num=-1;
    for(int i=0;i<len;i++){
        if('0'<=ipstr[i]&&ipstr[i]<='9'){
            if (num==0){
                //each piece should not start with 0, like 013 or 003; multi zero is not allowed, like 231.000.123.1
                return -1;
            }
            if (num==-1){
                num=0;
            }
            num=num*10+ipstr[i]-'0';
            if (num>255){
                return -1;
            };
            last=ipstr[i];
        } else if(ipstr[i]=='.'){
            if(num==-1){
                return -1;
            }
            counter++;
            *result=((*result)<<8)+num;
            num=-1;
            last=ipstr[i];
        } else if (ipstr[i]==' '){
            for(;i<len&&ipstr[i]==' ';i++){}
            if (i==len||
                '0'<=ipstr[i]&&ipstr[i]<='9'&&last!='.'||
                ipstr[i]=='.'&&!('0'<=last&&last<='9')){
                return -1;
            }
            i--;
        } else{
            return -1;
        }
    }
    if (num==-1||counter!=3){
        cerr<<"the final piece of num:"<<num<<", numbers of dot:"<<counter<<endl;
        return -1;
    }

    *result=((*result)<<8)+num;
    return 0;
}
