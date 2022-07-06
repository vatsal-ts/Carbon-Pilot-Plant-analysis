
#include<stdlib.h>
#include<iostream>

int main(){
    char st[]= "1.4 3.13";
    char *p;
    char* r;

    double t1=strtod(st,&p);
    double t2=strtod(p,NULL);

    printf("%.2f",t1+t2);
    return 0;


}