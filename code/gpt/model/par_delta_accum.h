#pragma once
#include "par_matrix.h"

using NCuda::TModelMatrix;

enum EAddToModel
{
    GRADIENT_ACCUMULATE,
    GRADIENT_APPLY,
};


class TMMDeltaAccumulate : public IMMDeltaHook
{
    TArray2D<float> DeltaAccum;
    bool HasAccumulatedDelta = false;
    EAddToModel AddToModel = GRADIENT_APPLY;
    TVector<bool> NonzeroRowFlag;
    TIntrusivePtr<TModelMatrix> ModelMatrix;

    void OnDelta() override;
public:
    TMMDeltaAccumulate(TIntrusivePtr<TModelMatrix> p) : ModelMatrix(p)
    {
    }
    void SetAddToModel(EAddToModel addToModel)
    {
        AddToModel = addToModel;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMMDeltaAccumulateGen : public IMMDeltaHookGen
{
    TVector<TIntrusivePtr<TMMDeltaAccumulate>> DeltaAccumArr;
    EAddToModel AddToModel = GRADIENT_APPLY;
    EAddToModel PrevAddToModel = GRADIENT_APPLY;

public: // public to allow chain calling from other hooks
    IMMDeltaHook *CreateDeltaHook(yint idx, TIntrusivePtr<TModelMatrix> p) override
    {
        TMMDeltaAccumulate *res = new TMMDeltaAccumulate(p);
        DeltaAccumArr.push_back(res);
        return res;
    }

    void OnIterationStart() override
    {
        if (AddToModel != PrevAddToModel) {
            for (TIntrusivePtr<TMMDeltaAccumulate> &p : DeltaAccumArr) {
                p->SetAddToModel(AddToModel);
            }
            PrevAddToModel = AddToModel;
        }
    }

public:
    void SetAddToModel(EAddToModel addToModel)
    {
        AddToModel = addToModel;
    }
};
