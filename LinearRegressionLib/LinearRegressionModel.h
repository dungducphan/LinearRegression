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
    virtual void SetTargetVector(Double_t* externalTarget);
    virtual void Train();
    virtual Double_t GetWeight(unsigned int idx);
    virtual Double_t GetWeightError(unsigned int idx);

    virtual Double_t CostFunction(Double_t* weights);
    virtual Double_t PredictTrainTarget(Double_t* weights, Double_t* features);

private:
    TMinuit* fMinimizer;

    unsigned int kNumberOfFeatures;
    unsigned int kNumberOfTrainingSamples;

    Double_t* kFeatureMatrix;
    Double_t* kTargetVector;
    Double_t* kWeightVector;
    Double_t* kWeightErrorVector;
};


#endif //LINEARREGRESSION_LINEARREGRESSIONMODEL_H
