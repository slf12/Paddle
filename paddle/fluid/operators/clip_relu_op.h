/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ClipReluForward {
  HOSTDEVICE ClipReluForward(const T* threshold) : threshold(threshold) {}

  HOSTDEVICE T operator()(const T& val) const {
    if (val > *threshold) {
      return *threshold;
    } else if (val < 0) {
      return val;
    } else {
      return val;
    }
  }

  const T* threshold;
};
template <typename DeviceContext, typename T>
class ClipReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    VLOG(5) << "compute in clip_relu";

    auto* in0 = ctx.Input<framework::LoDTensor>("X");
    auto* in1 = ctx.Input<framework::Tensor>("Threshold");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());
    VLOG(5) << "compute 1  in clip_relu";
    auto* place = ctx.template device_context<DeviceContext>().eigen_device();
    auto input = EigenVector<T>::Flatten(*in0);
    auto out = EigenVector<T>::Flatten(*output);
    VLOG(5) << "compute 2 in clip_relu";
    const T* threshold = in1->data<T>();
    VLOG(5) << "compute 3 in clip_relu";
    // out.device(place) =
    //   input.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
    out.device(*place) = input.unaryExpr(ClipReluForward<T>(threshold));
    VLOG(5) << "compute 4 in clip_relu";
  }
};

template <typename T>
struct ClipReluBackward {
  HOSTDEVICE ClipReluBackward(const T* threshold) : threshold(threshold) {}

  HOSTDEVICE T operator()(const T& val) const {
    if (val < *threshold) {
      return 0;
    } else {
      return 1;
    }
  }

  const T* threshold;
};
template <typename T>
struct ClipReluBackward1 {
  HOSTDEVICE ClipReluBackward1(const T* threshold) : threshold(threshold) {}

  HOSTDEVICE T operator()(const T& val) const {
    if (val < *threshold) {
      return 1;
    } else {
      return 0;
    }
  }

  const T* threshold;
};
template <typename DeviceContext, typename T>
class ClipReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* d_out_t =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* threshold_t = ctx.Input<framework::LoDTensor>("Threshold");
    auto* x_t = ctx.Input<framework::LoDTensor>("X");
    auto* d_x_t = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* d_threshold_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Threshold"));
    VLOG(5) << "compute grad 1  in clip_relu";
    auto* place = ctx.template device_context<DeviceContext>().eigen_device();
    auto d_out = EigenVector<T>::Flatten(*d_out_t);
    auto x = EigenVector<T>::Flatten(*x_t);
    const T* threshold = threshold_t->data<T>();

    // d_x_t->mutable_data<T>(ctx.GetPlace());
    // d_threshold_t->mutable_data<T>(ctx.GetPlace());
    VLOG(5) << "compute grad 2  in clip_relu";
    if (d_x_t) {
      d_x_t->mutable_data<T>(ctx.GetPlace());
      auto d_x = EigenVector<T>::Flatten(*d_x_t);

      VLOG(5) << "compute grad 4  in clip_relu";
      d_x.device(*place) =
          d_out * x.unaryExpr((ClipReluBackward1<T>(threshold)));
    }
    if (d_threshold_t) {
      framework::Tensor tmp_tensor;
      auto in_counts = x_t->numel();
      tmp_tensor.mutable_data<T>({static_cast<int>(in_counts)}, ctx.GetPlace());

      auto tmp = EigenVector<T>::Flatten(tmp_tensor);

      d_threshold_t->mutable_data<T>(ctx.GetPlace());

      auto d_threshold = EigenVector<T>::Flatten(*d_threshold_t);
      VLOG(5) << "compute grad 5  in clip_relu";
      tmp.device(*place) = x.unaryExpr(ClipReluBackward<T>(threshold));
      auto mat_dims =
          framework::make_ddim({static_cast<int>(x_t->dims()[0]),
                                static_cast<int>(in_counts / x_t->dims()[0])});
      auto errors_mat_view = EigenMatrix<T>::From(tmp_tensor, mat_dims);
      d_threshold.device(*place) =
          errors_mat_view.sum(Eigen::array<int, 1>({{1}}));
    }
  }
};
}  // namespace operators
}  // namespace paddle
