/*************************************************************************
  > File Name: TensorAccessor.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
NOTICE: This file is based on TensorAccessor of ATen
 ************************************************************************/

#ifndef TENSORACCESSOR_H
#define TENSORACCESSOR_H

#include<vector>

template<typename T, size_t N, typename index_t=int>
class TensorAccessorBase
{
    protected:
        const index_t* _sizes;
        const index_t* _strides;
        T* _data;
    public:
        TensorAccessorBase(
                T *data, 
                const index_t *sizes, 
                const index_t *strides)
            : _data(data),_sizes(sizes),_strides(strides) {};
        T data() {return _data;}
        const T data() const {return _data;}
        std::vector<index_t> sizes() const {
            return std::vector<index_t>(_sizes, _sizes+N);
        }
        std::vector<index_t> strides() const {
            return std::vector<index_t>(_strides, _strides+N);
        }
};

template<typename T, size_t N, typename index_t=int>
class TensorAccessor: public TensorAccessorBase<T,N,index_t>
{
    public:
        TensorAccessor(T* data, const index_t* sizes, const index_t* strides)
        : TensorAccessorBase<T,N,index_t>(data, sizes, strides){}
        TensorAccessor<T, N-1,index_t> operator[](index_t i) {
            return TensorAccessor<T,N-1,index_t>(this->_data + this->_strides[0]*i,
                    this->_sizes+1,this->_strides+1);
        }
        const TensorAccessor<T, N-1,index_t> operator[](index_t i) const {
            return TensorAccessor<T,N-1,index_t>(this->_data + this->_strides[0]*i,
                    this->_sizes+1,this->_strides+1);
        }
};

template<typename T, typename index_t>
class TensorAccessor<T,1,index_t>: public TensorAccessorBase<T,1,index_t>
{
    public:
        TensorAccessor(T* data, const index_t* sizes, const index_t* strides)
        : TensorAccessorBase<T,1,index_t>(data, sizes, strides){}
        T & operator[](index_t i) {
            return this->_data[this->_strides[0]*i];
        }
        const T & operator[](index_t i) const {
            return this->_data[this->_strides[0]*i];
        }
};

#endif // TENSORACCESSOR_H
