/*************************************************************************
  > File Name: test.cpp
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<ctime>
#include<cassert>
#ifdef USEATEN
#include<ATen/ATen.h>
#endif
#include "Tensor.h"
using std::cout; using std::endl; using std::ends;

int set_index(int L, int M, int N, double *in, 
        std::vector<std::vector<double*>> &out)
{
    out.resize(L);
    for (int i=0; i<L; ++i)
        out[i].resize(M);
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            out[i][j] = in+(i*M+j)*N;
    return 0;
}

int main(int argc, char* argv[])
{
    int L,M,N;
    if (argc<4) 
    {
        L=500; M=500; N=500;
    }
    else 
    {
        L = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    timespec start,end;
    bool flag = true;

// test naive array
    if (flag)
    {
    // array construct
    clock_gettime(CLOCK_MONOTONIC, &start);
    std::vector<double,default_init_allocator<double>> _b(L*M*N);
    std::vector<double,default_init_allocator<double>> _d(L*M*N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "array construct: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    double *b = _b.data();
    double *d = _d.data();
    // (OMP) array fill
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                b[(i*M+j)*N+k] = 1.1;
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) array fill: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) array initialize
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                b[(i*M+j)*N+k] = 1.0/(i+j+k+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) array initialize: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) array copy
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                d[(i*M+j)*N+k] = b[(i*M+j)*N+k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) array copy: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) array laplace
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                d[(i*M+j)*N+k] = b[((i+1)*M+j)*N+k]+b[((i-1)*M+j)*N+k]
                    +b[(i*M+j+1)*N+k]+b[(i*M+j-1)*N+k]
                    +b[(i*M+j)*N+k+1]+b[(i*M+j)*N+k-1]
                    -6*b[(i*M+j)*N+k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) array laplace: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    cout << endl;
    }

// test TensorAccessor
    if (flag)
    {
    Tensor<double,1> bT({10});
    Tensor<double,1> dT({10});
    TensorAccessor<double,1> b = bT.accessor();
    TensorAccessor<double,1> d = dT.accessor();
    bT.fill_(1.3);
    for (int i=0; i<b.sizes()[0]; ++i)
        assert(b[i] == 1.3);
    cout << "1.3 == " << b[b.sizes()[0]-1]
        << " " << b[b.sizes()[0]] 
        << endl;
    b.fill_(0.9);
    for (int i=0; i<b.sizes()[0]; ++i)
        assert(b[i] == 0.9);
    cout << "0.9 == " << b[b.sizes()[0]-1]
        << " " << b[b.sizes()[0]] 
        << endl;
    dT.copy_(bT);
    for (int i=0; i<b.sizes()[0]; ++i)
        assert(d[i] == 0.9);
    bT.fill_(1.3);
    d.copy_(b);
    for (int i=0; i<b.sizes()[0]; ++i)
        assert(d[i] == 1.3);
    // Tensor construct
    clock_gettime(CLOCK_MONOTONIC, &start);
    Tensor<double,3> aT({L,M,N});
    Tensor<double,3> cT({L,M,N});
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Tensor construct: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    TensorAccessor<double,3> a = aT.accessor();
    TensorAccessor<double,3> c = cT.accessor();
    TensorAccessor<const double,3> aConst = a.ConstAccessor();
    cout << "aConst[0][0][0] before fill_(1.1) " << aConst[0][0][0] << endl;
    // Tensor.fill
    clock_gettime(CLOCK_MONOTONIC, &start);
    aT.fill_(1.1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "aConst[0][0][0] after fill_(1.1) " << aConst[0][0][0] << endl;
    cout << "Tensor.fill: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) TensorAccessor.fill
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<c.sizes()[0]; ++i)
        c[i].fill_(1.2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) TensorAccessor.fill: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) TensorAccessor initialize
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                a[i][j][k] = 1.0/(i+j+k+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) TensorAccessor initialize: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // Tensor.copy
    clock_gettime(CLOCK_MONOTONIC, &start);
    cT.copy_(aT);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Tensor.copy: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) TensorAccessor.copy
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
                c[i].copy_(a[i]);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) TensorAccessor.copy: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) element TensorAccessor.copy
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                c[i][j][k] = a[i][j][k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) element TensorAccessor copy: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) TensorAccessor laplace
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                c[i][j][k] = a[i+1][j][k]+a[i-1][j][k]
                    +a[i][j+1][k]+a[i][j-1][k]
                    +a[i][j][k+1]+a[i][j][k-1]
                    -6*a[i][j][k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) TensorAccessor laplace: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    cout << "sizes: " << aT.sizes()[0] << " " << 
        aT.sizes()[1] << " " << 
        aT.sizes()[2] << endl;
    cout << "strides: " << aT.strides()[0] << " " << 
        aT.strides()[1] << " " << 
        aT.strides()[2] << endl;
    cout << endl;
    }

// test at::TensorAccessor
#ifdef USEATEN
    if (flag)
    {
    // at::TensorAccessor construct
    clock_gettime(CLOCK_MONOTONIC, &start);
    at::Tensor vT = at::empty({L,M,N}, at::kDouble);
    at::Tensor uT = at::empty({L,M,N}, at::kDouble);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "at::Tensor construct: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    at::TensorAccessor<double,3> v = vT.accessor<double,3>();
    at::TensorAccessor<double,3> u = uT.accessor<double,3>();
    // at::TensorAccessor initialize
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                u[i][j][k] = 1.0/(i+j+k+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) at::TensorAccessor initialize " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) at::TensorAccessor laplace
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                v[i][j][k] = u[i+1][j][k]+u[i-1][j][k]
                    +u[i][j+1][k]+u[i][j-1][k]
                    +u[i][j][k+1]+u[i][j][k-1]
                    -6*u[i][j][k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) at::TensorAccessor laplace: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    cout << endl;
    }
#endif
// test vector based tensor indexing
    if (flag)
    {
    // vector construc
    clock_gettime(CLOCK_MONOTONIC, &start);
    Tensor<double, 3> aT({L,M,N});
    Tensor<double, 3> cT({L,M,N});
    std::vector<std::vector<double*>> a;
    set_index(L,M,N,aT.data(),a);
    std::vector<std::vector<double*>> c;
    set_index(L,M,N,cT.data(),c);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "vector construct: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) vector initialize
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                a[i][j][k] = 1.0/(i+j+k+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) vector initialize: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    // (OMP) vector laplace
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for schedule(static)
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                c[i][j][k] = a[i+1][j][k]+a[i-1][j][k]
                    +a[i][j+1][k]+a[i][j-1][k]
                    +a[i][j][k+1]+a[i][j][k-1]
                    -6*a[i][j][k];
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "(OMP) vector laplace: " << 
        (double)((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9) 
        << endl;
    cout << endl;
    }
    return 0;
}
