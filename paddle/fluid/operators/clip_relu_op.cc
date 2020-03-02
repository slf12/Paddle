/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/clip_relu_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class ClipReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of ClipReluOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Threshold"), true,
                      platform::errors::NotFound(
                          "Input(Threshold) of ClipReluOp is not found."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output(Out) of ClipReluOp is not found."));
    VLOG(5) << "InferShape in clip_relu";

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    VLOG(5) << "get kernel type in clip_relu";
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};
class ClipReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of clip_relu operator, an N-D Tensor, "
             "with data type float32, float64.");
    AddInput("Threshold",
             "(Tensor), The threshold of clip_relu op with shape (1,1).");
    AddOutput(
        "Out",
        "Output of clip_relu operator, a Tensor with the same shape as input.");
    AddComment(R"DOC(
Relu6 Activation Operator.

$out = \min(\max(0, x), threshold)$

)DOC");
  }
};

class ClipReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of ClipReluOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Threshold"), true,
                      platform::errors::NotFound(
                          "Input(Threshold) of ClipReluOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::NotFound("Input(X) of ClipReluOp is not found."));
    auto x_dims = ctx->GetInputDim("X");

    auto threshold_dims = ctx->GetInputDim("Threshold");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    auto threshold_grad_name = framework::GradVarName("Threshold");
    if (ctx->HasOutput(threshold_grad_name)) {
      ctx->SetOutputDim(threshold_grad_name, threshold_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ClipReluGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("clip_relu_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Threshold", this->Input("Threshold"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Threshold"),
                  this->InputGrad("Threshold"));

    return op;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(clip_relu, ops::ClipReluOp, ops::ClipReluOpMaker,
                  ops::ClipReluGradMaker<paddle::framework::OpDesc>,
                  ops::ClipReluGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(clip_relu_grad, ops::ClipReluGradOp);

REGISTER_OP_CPU_KERNEL(
    clip_relu, ops::ClipReluKernel<paddle::platform::CPUDeviceContext, float>);
//  ops::ClipReluKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    clip_relu_grad,
    ops::ClipReluGradKernel<paddle::platform::CPUDeviceContext, float>);
// ops::ClipReluGradKernel<paddle::platform::CPUDeviceContext, double>);
