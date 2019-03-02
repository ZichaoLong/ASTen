/*************************************************************************
  > File Name: Tensor.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
 ************************************************************************/

#ifndef TENSOR_H
#define TENSOR_H

#include<memory>
#include<vector>
#include "TensorAccessor.h"

// A no-initialization allocator from:
// https://en.cppreference.com/w/cpp/container/vector/resize
// Allocator adaptor that interposes construct() calls to
// convert value initialization into default initialization.
template <typename T, typename A=std::allocator<T>>
class default_init_allocator : public A {
    typedef std::allocator_traits<A> a_t;
    public:
    template <typename U> struct rebind {
        using other =
            default_init_allocator<
            U, typename a_t::template rebind_alloc<U>
            >;
    };

    using A::A;

    template <typename U>
        void construct(U* ptr)
        noexcept(std::is_nothrow_default_constructible<U>::value) {
            ::new(static_cast<void*>(ptr)) U;
        }
    template <typename U, typename...Args>
        void construct(U* ptr, Args&&... args) {
            a_t::construct(static_cast<A&>(*this),
                    ptr, std::forward<Args>(args)...);
        }
};

// declaration
// TensorBase<T,N,index_t>
template<typename T, size_t N, typename index_t=int>
class TensorBase
{
    protected:
        const int _dim;
        std::vector<index_t> _sizes;
        std::vector<index_t> _strides;
        std::vector<T,default_init_allocator<T>> _data_vector;
        T *_data;
    public:
        TensorBase(T *data, const index_t *sizes, const index_t *strides)
            : _dim(N),_data(data),
            _sizes(sizes,sizes+N),
            _strides(strides,strides+N) {};
        TensorBase(T *data, const std::vector<index_t> sizes);
        TensorBase(const std::vector<index_t> sizes);

        TensorAccessor<T,N,index_t> accessor() const& {
            return TensorAccessor<T,N,index_t>(
                    _data, _sizes.data(), _strides.data());
        }
        TensorAccessor<T,N,index_t> accessor() && = delete;

        T *data() {return _data;}
        const T *data() const {return _data;}

        std::vector<index_t> sizes() const {
            return std::vector<index_t>(_sizes);
        }
        std::vector<index_t> strides() const {
            return std::vector<index_t>(_strides);
        }
        int view_as_continuous();
        int dim() {return _dim;}
};

// declaration
// Tensor<T,N,index_t> and Tensor<T,1,index_t>
template<typename T, size_t N, typename index_t=int>
class Tensor : public TensorBase<T,N,index_t>
{
    public:
        Tensor(T *data, const index_t *sizes, const index_t *strides)
            : TensorBase<T,N,index_t>(data, sizes, strides) {};
        Tensor(T *data, const std::vector<index_t> sizes)
            : TensorBase<T,N,index_t>(data, sizes) {};
        Tensor(const std::vector<index_t> sizes)
            : TensorBase<T,N,index_t>(sizes) {};

        void fill_(T v);
        template<typename T_, typename index_t_>
            void copy_(const Tensor<T_,N,index_t_> &another);
};
template<typename T, typename index_t>
class Tensor<T,1,index_t> : public TensorBase<T,1,index_t>
{
    public:
        Tensor(T *data, const index_t *sizes, const index_t *strides)
            : TensorBase<T,1,index_t>(data, sizes, strides) {};
        Tensor(T *data, const std::vector<index_t> sizes)
            : TensorBase<T,1,index_t>(data, sizes) {};
        Tensor(const std::vector<index_t> sizes)
            : TensorBase<T,1,index_t>(sizes) {};

        void fill_(T v);
        template<typename T_, typename index_t_>
            void copy_(const Tensor<T_,1,index_t_> &another);
};


// definition
// TensorBase<T,N,index_t> constructor
template<typename T, size_t N, typename index_t>
int TensorBase<T,N,index_t>::view_as_continuous()
{
    _strides.resize(N);
    _strides[N-1] = 1;
    for (int i=(int)N-2; i>-1; --i)
        _strides[i] = _strides[i+1]*_sizes[i+1];
    return 0;
}
template<typename T, size_t N, typename index_t>
TensorBase<T,N,index_t>::TensorBase(T *data, const std::vector<index_t> sizes)
    : _dim(N),_data(data),_sizes(sizes)
{
    view_as_continuous();
}
template<typename T, size_t N, typename index_t>
TensorBase<T,N,index_t>::TensorBase(const std::vector<index_t> sizes)
    : _dim(N)
{
    _sizes = sizes;
    view_as_continuous();
    size_t L = 1;
    for (int i=0; i<N; ++i)
        L *= _sizes[i];
    _data_vector.resize(L);
    _data_vector.shrink_to_fit();
    _data = _data_vector.data();
}


// definition
// Tensor<T,N,index_t> and Tensor<T,1,index_t> utils
template<typename T, size_t N, typename index_t>
void Tensor<T,N,index_t>::fill_(T v)
{
#pragma omp parallel for
    for (index_t i=0; i<this->_sizes[0]; ++i)
        this->accessor().operator[](i).fill_(v);
}
template<typename T, size_t N, typename index_t>
template<typename T_, typename index_t_>
void Tensor<T,N,index_t>::copy_(
        const Tensor<T_,N,index_t_> &another)
{
    index_t I = std::min(this->_sizes[0], (index_t) another._sizes[0]);
#pragma omp parallel for
    for (index_t i=0; i<I; ++i)
        this->accessor().operator[](i).copy_(
                another.accessor().operator[](i));
}

template<typename T, typename index_t>
void Tensor<T,1,index_t>::fill_(T v)
{
#pragma omp parallel for
    for (index_t i=0; i<this->_sizes[0]; ++i)
        this->_data[i*this->_strides[0]] = v;
}
template<typename T, typename index_t>
template<typename T_, typename index_t_>
void Tensor<T,1,index_t>::copy_(
        const Tensor<T_,1,index_t_> &another)
{
    index_t I = std::min(this->_sizes[0], (index_t) another._sizes[0]);
#pragma omp parallel for
    for (index_t i=0; i<I; ++i)
        this->_data[i*this->_strides[0]] = another._data[i*another._strides[0]];
}
#endif // TENSOR_H
