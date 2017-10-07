//
// Created by Dung Phan on 10/6/17.
//

#include "LinearRegressor.h"

LinearRegressor::LinearRegressor() {
    kLRData  = new LinearRegressionData();
    kLRModel = new LinearRegressionModel();
}

LinearRegressor::~LinearRegressor() = default;

void LinearRegressor::SetNumberOfFeatures(unsigned int n) {
    kLRData->SetNumberOfFeatures(n);
    kLRModel->SetNumberOfFeatures(n);
}

void LinearRegressor::SetNumberOfSamples(unsigned int n) {
    kLRData->SetNumberOfSamples(n);
}

void LinearRegressor::SetFeatureScaling(bool doFeatureScale) {
    kLRData->SetFeatureScaling(doFeatureScale);
}

void LinearRegressor::ImportFeature(Double_t *featureColumn, unsigned int featureIdx) {
    kLRData->ImportFeature(featureColumn, featureIdx);
}

void LinearRegressor::ImportTarget(Double_t *targetColumn) {
    kLRData->ImportTarget(targetColumn);
}

void LinearRegressor::SetTestSampleRatio(double testRatio) {
    kLRData->SetTestSampleRatio(testRatio);
}

void LinearRegressor::DivideTrainTest() {
    kLRData->DivideTrainTest();
}

void LinearRegressor::Train() {
    kLRModel->SetNumberOfTrainingSamples(kLRData->GetNumberOfTrainSamples());
    kLRModel->SetFeatureMatrix(kLRData->GetTrainFeatureMatrix());
    kLRModel->SetTarget(kLRData->GetTrainTarget());
    kLRModel->Train();
}

Double_t LinearRegressor::GetWeight(unsigned int weightIdx) {
    if (!kLRModel->kIsTrained) {
        Train();
    }
    return kLRModel->GetWeight(weightIdx);
}

Double_t LinearRegressor::GetRSquared() {
    if (!kLRModel->kIsTrained) {
        Train();
    }
    return kLRModel->GetRSquared();
}

Double_t LinearRegressor::GetAdjustedRSquared() {
    if (!kLRModel->kIsTrained) {
        Train();
    }
    return kLRModel->GetAdjustedRSquared();
}

void LinearRegressor::Test() {
    if (!kLRModel->kIsTrained) {
        Train();
    }
    unsigned int n = kLRData->GetNumberOfFeatures();

    Double_t* trainFeatures          = kLRData->GetTrainFeatureMatrix();
    Double_t* trainPredictedTarget   = kLRData->GetTrainPredictedTarget();
    for (unsigned int i = 0; i < kLRData->GetNumberOfTrainSamples(); i++) {
        *(trainPredictedTarget + i) = kLRModel->TrainedPredictor(trainFeatures + i * n);
    }

    Double_t* testFeatures          = kLRData->GetTestFeatureMatrix();
    Double_t* testPredictedTarget   = kLRData->GetTestPredictedTarget();
    for (unsigned int i = 0; i < kLRData->GetNumberOfTestSamples(); i++) {
        *(testPredictedTarget + i) = kLRModel->TrainedPredictor(testFeatures + i * n);
    }
}

void LinearRegressor::PrintTrainTarget() {
    Double_t* trainTarget            = kLRData->GetTrainTarget();
    Double_t* trainPredictedTarget   = kLRData->GetTrainPredictedTarget();

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(30) << "Truth Train Target";
    std::cout << "\t";
    std::cout << std::setw(30) << "Predicted Train Target";
    std::cout << std::endl;
    std::cout << std::endl;
    for (unsigned int i = 0; i < kLRData->GetNumberOfTrainSamples(); i++) {
        std::cout << std::setw(30) << std::fixed << std::setprecision(3) << *(trainTarget + i);
        std::cout << "\t";
        std::cout << std::setw(30) << std::fixed << std::setprecision(3) << *(trainPredictedTarget + i);
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void LinearRegressor::PrintTestTarget() {
    Double_t* testTarget            = kLRData->GetTestTarget();
    Double_t* testPredictedTarget   = kLRData->GetTestPredictedTarget();

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(30) << "Truth Test Target";
    std::cout << "\t";
    std::cout << std::setw(30) << "Predicted Test Target";
    std::cout << std::endl;
    std::cout << std::endl;
    for (unsigned int i = 0; i < kLRData->GetNumberOfTestSamples(); i++) {
        std::cout << std::setw(30) << std::fixed << std::setprecision(3) << *(testTarget + i);
        std::cout << "\t";
        std::cout << std::setw(30) << std::fixed << std::setprecision(3) << *(testPredictedTarget + i);
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

Double_t *LinearRegressor::GetTrainPredictedTarget() {
    return kLRData->GetTrainPredictedTarget();
}

Double_t *LinearRegressor::GetTestPredictedTarget() {
    return kLRData->GetTestPredictedTarget();
}

Double_t *LinearRegressor::GetTrainTarget() {
    return kLRData->GetTrainTarget();
}

Double_t *LinearRegressor::GetTestTarget() {
    return kLRData->GetTestTarget();
}

Double_t *LinearRegressor::GetTarget() {
    return kLRData->GetTarget();
}

Double_t *LinearRegressor::GetFeatureMatrix() {
    return kLRData->GetFeatureMatrix();
}

Double_t *LinearRegressor::GetTrainFeatureMatrix() {
    return kLRData->GetTrainFeatureMatrix();
}

Double_t *LinearRegressor::GetTestFeatureMatrix() {
    return kLRData->GetTestFeatureMatrix();
}

unsigned int LinearRegressor::GetNumberOfFeatures() {
    return kLRData->GetNumberOfFeatures();
}

unsigned int LinearRegressor::GetNumberOfSamples() {
    return kLRData->GetNumberOfSamples();
}

unsigned int LinearRegressor::GetNumberOfTrainSamples() {
    return kLRData->GetNumberOfTrainSamples();
}

unsigned int LinearRegressor::GetNumberOfTestSamples() {
    return kLRData->GetNumberOfTestSamples();
}

Double_t *LinearRegressor::GetTrainFeature(unsigned int colIdx) {
    return kLRData->GetTrainFeature(colIdx);
}

Double_t *LinearRegressor::GetTestFeature(unsigned int colIdx) {
    return kLRData->GetTestFeature(colIdx);
}

