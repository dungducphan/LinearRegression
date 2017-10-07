//
// Created by Dung Phan on 10/6/17.
//

#ifndef LINEARREGRESSION_LINEARREGRESSIONDATA_H
#define LINEARREGRESSION_LINEARREGRESSIONDATA_H

#include <TMath.h>

#include <Rtypes.h>
#include <random>
#include <iostream>
#include <iomanip>

class LinearRegressionData {
public:
    LinearRegressionData();
    virtual ~LinearRegressionData();

    virtual void ImportFeature(Double_t* featureColumn, unsigned int featureIdx);
    virtual void ImportTarget(Double_t* targetColumn);

    virtual void SetNumberOfFeatures(unsigned int n);
    virtual void SetNumberOfSamples(unsigned int n);
    virtual void SetTestSampleRatio(double testRatio);
    virtual void SetFeatureScaling(bool doScalingFeatures);

    virtual void DivideTrainTest();

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

    virtual void FeatureScaleThisSample(Double_t *sampleFeatures);

private:
    double kTestRatio;

    /*
     * Set kFeatureScaling to true to normalize the feature column.
     */
    bool kFeatureScaling;

    /*
     * Internal check if data storage has been malloc-ed.
     */
    bool kIsMalloced;

    /*
     * Features from data. Scaled if (kFeatureScaling == true).
     */
    Double_t* kFeatureMatrix;
    Double_t* kTrainFeatureMatrix;
    Double_t* kTestFeatureMatrix;
    Double_t* kTrainFeatureMatrixTranspose;
    Double_t* kTestFeatureMatrixTranspose;

    /*
     * Ground-truth targets from data.
     */
    Double_t* kTarget;
    Double_t* kTrainTarget;
    Double_t* kTestTarget;

    /*
     * Predicted targets from the trained model.
     */
    Double_t* kTrainPredictedTarget;
    Double_t* kTestPredictedTarget;

    /*
     * If features are scaled. The mean and stddev from each feature column is saved here.
     */
    Double_t* kFeatureMeans;
    Double_t* kFeatureStdDev;

    /*
     * Data dimension.
     * kNumberOfFeatures is number of columns in kFeatureMatrix.
     * kNumberOfSamples if the number of rows in kFeatureMatrix and kTarget.
     */
    unsigned int kNumberOfFeatures;
    unsigned int kNumberOfSamples;
    unsigned int kNumberOfTrainSamples;
    unsigned int kNumberOfTestSamples;

    /*
     * Scale feature column before importing.
     */
    virtual void ScaleFeature(Double_t* featureColumn, unsigned int featureIdx);
    virtual Double_t GetMeanOfFeature(Double_t* featureColumn);
    virtual Double_t GetStdDevOfFeature(Double_t* featureColumn);
    virtual Double_t ReScale(Double_t feature, unsigned int featureIdx);

    /*
     * Malloc for member data that stores the feature matrices and targets.
     */
    virtual void MallocStorage();

    /*
     * Randomly choose a row (a sample) data to import into test data.
     */
    virtual bool IsTakenIntoTestSample(unsigned int chooseN, unsigned int amongAll);
};


#endif //LINEARREGRESSION_LINEARREGRESSIONDATA_H
