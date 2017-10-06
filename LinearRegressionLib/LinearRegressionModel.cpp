//
// Created by Dung Phan on 10/5/17.
//

#include "LinearRegressionModel.h"

LinearRegressionModel::LinearRegressionModel() = default;

LinearRegressionModel::~LinearRegressionModel() {
    free(kWeightVector);
    free(kWeightErrorVector);
    free(kFeatureMatrix);
    free(kTargetVector);
}

void LinearRegressionModel::SetFeatureMatrix(Double_t* externalFeatureMatrix) {
    kFeatureMatrix = externalFeatureMatrix;
}

void LinearRegressionModel::SetTargetVector(Double_t* externalTarget) {
    kTargetVector = externalTarget;
}

Double_t LinearRegressionModel::PredictTrainTarget(Double_t *weights, Double_t *features) {
    Double_t predictedTarget = 0;
    for (int i = 0; i < kNumberOfFeatures; i++) {
        predictedTarget += features[i] * weights[i];
    }

    return predictedTarget;
}

Double_t LinearRegressionModel::CostFunction(Double_t* weights) {
    Double_t cost = 0.;
    for (unsigned int i = 0; i < kNumberOfTrainingSamples; i++) {
        cost += TMath::Power(PredictTrainTarget(weights, &kFeatureMatrix[i * kNumberOfFeatures]) - kTargetVector[i], 2);
    }

    return cost / (2 * (Double_t)kNumberOfTrainingSamples);
}

void LinearRegressionModel::Train() {
    fMinimizer = new TMinuit(kNumberOfFeatures);
    fMinimizer->SetObjectFit(this);
    fMinimizer->SetFCN(fcn);
    fMinimizer->SetPrintLevel(1);

    Int_t ierflg = 0;
    for (int i = 0; i < kNumberOfFeatures; i++) {
        std::string parName = "weight_" + std::to_string(i);
        fMinimizer->mnparm(i, parName.c_str(), 0, 0.01, 0, 0, ierflg);
    }

    // Start minimizing cost function
    fMinimizer->SetErrorDef(1);
    fMinimizer->SetMaxIterations(1500);
    fMinimizer->Migrad();

    // Get weights
    kWeightVector = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));
    kWeightErrorVector = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));
    for (int i = 0; i < kNumberOfFeatures; i++) {
        Double_t paramValue = 0;
        Double_t paramError = 0;
        fMinimizer->GetParameter(i, paramValue, paramError);
        kWeightVector[i] = paramValue;
        kWeightErrorVector[i] = paramError;
    }
}

void LinearRegressionModel::SetNumberOfTrainingSamples(unsigned int n) {
    kNumberOfTrainingSamples = n;
    kFeatureMatrix = (Double_t*) malloc(kNumberOfTrainingSamples * kNumberOfFeatures * sizeof(Double_t));
}

void LinearRegressionModel::SetNumberOfFeatures(unsigned int n) {
    kNumberOfFeatures = n;
    kTargetVector = (Double_t*) malloc(kNumberOfTrainingSamples * 1 * sizeof(Double_t));
}

Double_t LinearRegressionModel::GetWeight(unsigned int idx) {
    return kWeightVector[idx];
}

Double_t LinearRegressionModel::GetWeightError(unsigned int idx) {
    return kWeightErrorVector[idx];
}

void fcn(Int_t&, Double_t*, Double_t& cost, Double_t* weights, Int_t) {
    LinearRegressionModel* lrm = (LinearRegressionModel*) gMinuit->GetObjectFit();
    cost = lrm->CostFunction(weights);
}
