/*************************************************************************
  > File Name: test.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<omp.h>
#include "TensorAccessor.h"
using std::cout; using std::endl; using std::ends;


int main( )
{
    int L=500,M=200,N=300;
    double *_a = new double[L*M*N];
    double *_b = new double[L*M*N];
    double *b = _b;
    int sizes[] = {L,M,N};
    int strides[] = {M*N,N,1};
    TensorAccessor<double, 3, int> a(_a, sizes, strides);
    double start,end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                a[i][j][k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for TensorAccessor: " << end-start << endl;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                b[(i*M+j)*N+k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for naive array: " << end-start << endl;
    double sum = 0;
    for (int i=0; i<L*M*N; ++i)
        sum += std::abs(_a[i]-_b[i]);
    cout << "err: " << sum << endl;
    delete _a;
    delete _b;
    return 0;
}
