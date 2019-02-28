/*************************************************************************
  > File Name: Tensor.h
  > Author: Long Zichao
  > Mail: zlong@pku.edu.cn
  > Created Time: 2019-02-28
 ************************************************************************/

#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include "TensorAccessor.h"

template<typename T, size_t N, typename index_t=int>
class Tensor
{
    private:
        const int _dim;
        std::vector<index_t> _sizes;
        std::vector<index_t> _strides;
        std::vector<T> _data_vector;
        T *_data;
    public:
        Tensor(T *data, const index_t *sizes, const index_t *strides)
            : _dim(N),_data(data),
            _sizes(sizes,sizes+N),
            _strides(strides,strides+N) {};
        Tensor(T *data, const std::vector<index_t> sizes);
        Tensor(const std::vector<index_t> sizes);

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
template<typename T, size_t N, typename index_t>
int Tensor<T,N,index_t>::view_as_continuous()
{
    _strides.resize(N);
    _strides[N-1] = 1;
    for (int i=N-2; i>-1; --i)
        _strides[i] = _strides[i+1]*_sizes[i+1];
    return 0;
}
template<typename T, size_t N, typename index_t>
Tensor<T,N,index_t>::Tensor(T *data, const std::vector<index_t> sizes)
    : _dim(N),_data(data),_sizes(sizes)
{
    view_as_continuous();
}
template<typename T, size_t N, typename index_t>
Tensor<T,N,index_t>::Tensor(const std::vector<index_t> sizes)
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


#endif // TENSOR_H
