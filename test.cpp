/*************************************************************************
  > File Name: test.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<omp.h>
#ifdef USEATEN
#include<ATen/ATen.h>
#endif
#include "TensorAccessor.h"
using std::cout; using std::endl; using std::ends;


int main( )
{
    int L=500,M=500,N=500;
    std::vector<double> __a(L*M*N);
    std::vector<double> __b(L*M*N);
    std::vector<double> __c(L*M*N);
    std::vector<double> __d(L*M*N);
    double *b = __b.data();
    double *d = __d.data();
    double *_a = __a.data();
    double *_c = __c.data();
    d[0] = 10;
    int sizes[] = {L,M,N};
    int strides[] = {M*N,N,1};
    TensorAccessor<double, 3> a(_a, sizes, strides);
    TensorAccessor<double, 3> c(_c, sizes, strides);
#ifdef USEATEN
    at::Tensor vT = at::ones({L,M,N}, at::kDouble);
    at::Tensor uT = at::ones({L,M,N}, at::kDouble);
    at::TensorAccessor<double,3> v = vT.accessor<double,3>();
    at::TensorAccessor<double,3> u = uT.accessor<double,3>();
#endif
    double start,end;
    //
#ifdef USEATEN
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                u[i][j][k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for at::TensorAccessor: " << end-start << endl;
#endif
    //
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                a[i][j][k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for TensorAccessor: " << end-start << endl;
    //
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
        sum += std::abs(__a[i]-__b[i]);
    cout << "err: " << sum << endl;
    //
#ifdef USEATEN
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                v[i][j][k] = u[i+1][j][k]+u[i-1][j][k]
                    +u[i][j+1][k]+u[i][j-1][k]
                    +u[i][j][k+1]+u[i][j][k-1]
                    -6*u[i][j][k];
    end = omp_get_wtime();
    cout << "laplace for at::TensorAccessor: " << end-start << endl;
#endif
    //
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                c[i][j][k] = a[i+1][j][k]+a[i-1][j][k]
                    +a[i][j+1][k]+a[i][j-1][k]
                    +a[i][j][k+1]+a[i][j][k-1]
                    -6*a[i][j][k];
    end = omp_get_wtime();
    cout << "laplace for TensorAccessor: " << end-start << endl;
    //
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                d[(i*M+j)*N+k] = b[((i+1)*M+j)*N+k]+b[((i-1)*M+j)*N+k]
                    +b[(i*M+j+1)*N+k]+b[(i*M+j-1)*N+k]
                    +b[(i*M+j)*N+k+1]+b[(i*M+j)*N+k-1]
                    -6*b[(i*M+j)*N+k];
    end = omp_get_wtime();
    cout << "laplace for naive array: " << end-start << endl;
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                sum += std::abs(__c[(i*M+j)*N+k]-__d[(i*M+j)*N+k]);
    cout << "err: " << sum << endl;
    return 0;
}
