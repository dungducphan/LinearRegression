//
// Created by Dung Phan on 10/5/17.
//

#ifndef LINEARREGRESSION_LINEARREGRESSIONMODEL_H
#define LINEARREGRESSION_LINEARREGRESSIONMODEL_H

#include <TMinuit.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TMath.h>

#include <cstring>
#include <cstdlib>
#include <iostream>

static void fcn(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t);

class LinearRegressionModel : public TObject {
public:
    LinearRegressionModel();
    ~LinearRegressionModel() override;

    virtual void SetNumberOfFeatures(unsigned int n);
    virtual void SetNumberOfTrainingSamples(unsigned int n);
    virtual void SetFeatureMatrix(Double_t* externalFeatureMatrix);
    virtual void SetTarget(Double_t *externalTarget);
    virtual void Train();

    virtual Double_t GetWeight(unsigned int idx);
    virtual Double_t GetWeightError(unsigned int idx);

    virtual Double_t GetRSquared();
    virtual Double_t GetAdjustedRSquared();

    virtual Double_t CostFunction(Double_t* weights);
    virtual Double_t PredictTrainTarget(Double_t* weights, Double_t* features);
    virtual Double_t TrainedPredictor(Double_t *features);

    bool kIsTrained;

private:
    unsigned int kNumberOfFeatures;
    unsigned int kNumberOfTrainingSamples;

    Double_t* kFeatureMatrix;
    Double_t* kTarget;
    Double_t* kWeight;
    Double_t* kWeightError;

    /*
     * Model robustness metrics.
     */
    Double_t kTargetMean;
    Double_t kTotalSumOfSquares;
    Double_t kResidualSumOfSquares;
    Double_t kRSquared;
    Double_t kAdjustedRSquared;
    bool     kIsTargetMeanCalculated;
    bool     kIsTotalSumOfSquaresCalculated;
    bool     kIsResidualSumOfSquaresCalculated;
    bool     kIsRSquaredCalculated;
    bool     kIsAdjustedRSquaredCalculated;
    virtual void CalculateMeanTarget();
    virtual void CalculateTotalSumOfSquares();
    virtual void CalculateResidualSumOfSquares();
    virtual void CalculateRSquared();
    virtual void CalculateAdjustedRSquared();
};


#endif //LINEARREGRESSION_LINEARREGRESSIONMODEL_H
