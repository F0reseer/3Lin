#pragma once


struct TTrainingStep
{
    float Rate = 0.1f;
    float L2Reg = 0;

    TTrainingStep() {}
    TTrainingStep(float rate, float l2reg) : Rate(rate), L2Reg(l2reg) {}
    float GetShrinkMult() const
    {
        return 1 - Rate * L2Reg;
    }
    void ScaleRate(float x)
    {
        Rate *= x;
    }
};
