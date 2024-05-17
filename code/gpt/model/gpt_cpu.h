#pragma once
#include "model_params.h"
#include "model.h"

namespace NCPU_GPT
{
TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount);
}
