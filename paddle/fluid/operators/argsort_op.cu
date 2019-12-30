/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/argsort_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<paddle::platform::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t,
                 paddle::platform::float16> {};
}  // namespace cub

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Iter for move to next row
struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

template <typename T>
static __global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

// Sort by flag descending, True: descending. False: Ascending.
// Default is false.
template <typename T, typename IndType>
void ArgFullSort(const platform::CUDADeviceContext& ctx, const Tensor* input,
                 Tensor* output, Tensor* indices, const IndType num_rows,
                 const IndType num_cols, const bool descending) {
  auto cu_stream = ctx.stream();

  Tensor input_indices;

  const std::vector<IndType> dims = {num_rows, num_cols};
  auto dim = framework::make_ddim(dims);
  input_indices.Resize(dim);
  input_indices.mutable_data<IndType>(ctx.GetPlace());

  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](IndType col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };

  int block_size = ComputeBlockSize(num_cols);

  int maxGridDimX = ctx.GetCUDAMaxGridDimSize().x;
  // actually, int num_rows < max_grid_size
  int grid_size = num_rows < maxGridDimX ? num_rows : maxGridDimX;
  // Init a index array
  FillIndex<<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<IndType>(), num_rows, num_cols);

  T* sorted_out_ptr;
  IndType* sorted_indices_ptr;

  const T* inp = input->data<T>();
  T* out = output->mutable_data<T>(ctx.GetPlace());
  IndType* ind = indices->mutable_data<IndType>(ctx.GetPlace());

  sorted_out_ptr = out;
  sorted_indices_ptr = ind;

  // create iter for counting input
  cub::CountingInputIterator<IndType> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<IndType, SegmentOffsetIter,
                              cub::CountingInputIterator<IndType>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  cudaError_t err;
  if (descending) {
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, inp, sorted_out_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
  } else {
    err = cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes, inp, sorted_out_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      err,
      "ArgSortOP failed as could not launch "
      "cub::DeviceSegmentedRadixSort::SortPairsDescending to calculate"
      "temp_storage_bytes, status:%s.",
      temp_storage_bytes, cudaGetErrorString(err));

  Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  if (descending) {
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(), temp_storage_bytes, inp, sorted_out_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
  } else {
    err = cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data<uint8_t>(), temp_storage_bytes, inp, sorted_out_ptr,
        input_indices.data<IndType>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
  }

  PADDLE_ENFORCE_CUDA_SUCCESS(
      err,
      "ArgSortOP failed as could not launch "
      "cub::DeviceSegmentedRadixSort::SortPairsDescending to sort input, "
      "temp_storage_bytes:%d status:%s.",
      temp_storage_bytes, cudaGetErrorString(err));
}

template <typename T>
class ArgsortOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    int axis = ctx.Attr<int>("axis");
    bool descending = ctx.Attr<bool>("descending");

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    int64_t numel = input->numel();
    int64_t groups = numel / in_dims[axis];

    // Special case for full sort, speedup ~190x.
    if (axis == -1 || axis + 1 == in_dims.size()) {
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];
      const auto& dev_ctx = ctx.cuda_device_context();
      ArgFullSort<T, int64_t>(dev_ctx, input, output, indices, input_height,
                              input_width, descending);
    } else {
      // if not full sort, do transpose first
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.push_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.push_back(i);
      }
      trans.push_back(axis);
      framework::DDim trans_dims(in_dims);
      for (int i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }

      Tensor trans_inp;
      T* trans_inp_data = trans_inp.mutable_data<T>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      const auto& dev_ctx = ctx.cuda_device_context();
      // Do transpose
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_inp, trans);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      Tensor tmp_out;
      tmp_out.mutable_data<T>(trans_dims, ctx.GetPlace());
      T* out_data = output->mutable_data<T>(ctx.GetPlace());

      Tensor tmp_indices;
      // temp indices for sorting
      tmp_indices.mutable_data<int64_t>(trans_dims, ctx.GetPlace());
      indices->mutable_data<int64_t>(ctx.GetPlace());

      ArgFullSort<T, int64_t>(dev_ctx, &trans_inp, &tmp_out, &tmp_indices,
                              input_height, input_width, descending);

      TransCompute<platform::CUDADeviceContext, int64_t>(
          ndims, dev_ctx, tmp_indices, indices, trans);
      // transpose back
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, tmp_out,
                                                   output, trans);
      return;
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    argsort, paddle::operators::ArgsortOpCUDAKernel<float>,
    paddle::operators::ArgsortOpCUDAKernel<double>,
    paddle::operators::ArgsortOpCUDAKernel<int>,
    paddle::operators::ArgsortOpCUDAKernel<int64_t>,
    paddle::operators::ArgsortOpCUDAKernel<paddle::platform::float16>);
