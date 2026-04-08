#ifndef AI_ACCELERATOR_SDK_INFERENCE_ENGINE_H
#define AI_ACCELERATOR_SDK_INFERENCE_ENGINE_H

#include "tensor.h"
#include "model_loader.h"
#include <memory>

namespace ai_accelerator_sdk {

class InferenceEngine {
public:
    InferenceEngine(std::shared_ptr<Model> model);
    Tensor run(const Tensor& input);

private:
    std::shared_ptr<Model> model_;
};

} // namespace ai_accelerator_sdk

#endif // AI_ACCELERATOR_SDK_INFERENCE_ENGINE_H
