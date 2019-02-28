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
    if (argc<2) 
    {
        L=500; M=500; N=500;
    }
    else 
    {
        assert(argc>3);
        L = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    double start,end;
    bool flag = true;
// test TensorAccessor
    if (flag)
    {
    Tensor<double, 3> aT({L,M,N});
    Tensor<double, 3> cT({L,M,N});
    cout << "sizes: " << aT.sizes()[0] << " " << 
        aT.sizes()[1] << " " << 
        aT.sizes()[2] << endl;
    cout << "strides: " << aT.strides()[0] << " " << 
        aT.strides()[1] << " " << 
        aT.strides()[2] << endl;
    TensorAccessor<double, 3> a = aT.accessor();
    TensorAccessor<double, 3> c = cT.accessor();
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
    for (int i=1; i<L-1; ++i)
        for (int j=1; j<M-1; ++j)
            for (int k=1; k<N-1; ++k)
                c[i][j][k] = a[i+1][j][k]+a[i-1][j][k]
                    +a[i][j+1][k]+a[i][j-1][k]
                    +a[i][j][k+1]+a[i][j][k-1]
                    -6*a[i][j][k];
    end = omp_get_wtime();
    cout << "laplace for TensorAccessor: " << end-start << endl;
    }
// test at::TensorAccessor
#ifdef USEATEN
    if (flag)
    {
    at::Tensor vT = at::ones({L,M,N}, at::kDouble);
    at::Tensor uT = at::ones({L,M,N}, at::kDouble);
    at::TensorAccessor<double,3> v = vT.accessor<double,3>();
    at::TensorAccessor<double,3> u = uT.accessor<double,3>();
    double start,end;
    //
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                u[i][j][k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for at::TensorAccessor: " << end-start << endl;
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
    }
#endif
// test naive array
    if (flag)
    {
    std::vector<double> _b(L*M*N);
    std::vector<double> _d(L*M*N);
    double *b = _b.data();
    double *d = _d.data();
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                b[(i*M+j)*N+k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for naive array: " << end-start << endl;
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
    }
// test vector based tensor indexing
    if (flag)
    {
    Tensor<double, 3> aT({L,M,N});
    Tensor<double, 3> cT({L,M,N});
    std::vector<std::vector<double*>> a;
    set_index(L,M,N,aT.data(),a);
    std::vector<std::vector<double*>> c;
    set_index(L,M,N,cT.data(),c);
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i=0; i<L; ++i)
        for (int j=0; j<M; ++j)
            for (int k=0; k<N; ++k)
                a[i][j][k] = 1.0/(i+j+k+1);
    end = omp_get_wtime();
    cout << "time for vectorAccessor: " << end-start << endl;
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
    cout << "laplace for vectorAccessor: " << end-start << endl;
    }
    return 0;
}
