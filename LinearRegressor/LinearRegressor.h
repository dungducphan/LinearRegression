//
// Created by Dung Phan on 10/6/17.
//

#ifndef LINEARREGRESSION_LINEARREGRESSOR_H
#define LINEARREGRESSION_LINEARREGRESSOR_H

#include <LinearRegressionModel.h>
#include <LinearRegressionData.h>

#include <iomanip>

class LinearRegressor {
public:
    LinearRegressor();
    virtual ~LinearRegressor();

    virtual void SetNumberOfFeatures(unsigned int n);
    virtual void SetNumberOfSamples(unsigned int n);

    virtual void SetFeatureScaling(bool doFeatureScale);
    virtual void ImportFeature(Double_t* featureColumn, unsigned int featureIdx);
    virtual void ImportTarget(Double_t* targetColumn);

    virtual void SetTestSampleRatio(double testRatio);
    virtual void DivideTrainTest();

    virtual void Train();
    virtual Double_t GetWeight(unsigned int weightIdx);
    virtual Double_t GetRSquared();
    virtual Double_t GetAdjustedRSquared();

    virtual void Test();
    virtual void PrintTrainTarget();
    virtual void PrintTestTarget();

    /*
     * GET methods
     */
    virtual Double_t* GetTrainPredictedTarget();
    virtual Double_t* GetTestPredictedTarget();
    virtual Double_t* GetTrainTarget();
    virtual Double_t* GetTestTarget();
    virtual Double_t* GetTarget();

    virtual Double_t* GetFeatureMatrix();
    virtual Double_t* GetTrainFeatureMatrix();
    virtual Double_t* GetTestFeatureMatrix();

    virtual Double_t* GetTrainFeature(unsigned int colIdx);
    virtual Double_t* GetTestFeature(unsigned int colIdx);

    virtual unsigned int GetNumberOfFeatures();
    virtual unsigned int GetNumberOfSamples();
    virtual unsigned int GetNumberOfTrainSamples();
    virtual unsigned int GetNumberOfTestSamples();

private:
    LinearRegressionData*   kLRData;
    LinearRegressionModel*  kLRModel;
};


#endif //LINEARREGRESSION_LINEARREGRESSOR_H
