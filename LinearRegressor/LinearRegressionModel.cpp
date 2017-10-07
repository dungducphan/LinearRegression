//
// Created by Dung Phan on 10/5/17.
//

#include "LinearRegressionModel.h"

LinearRegressionModel::LinearRegressionModel() {
    kIsTrained = false;

    kIsTargetMeanCalculated = false;
    kIsTotalSumOfSquaresCalculated = false;
    kIsResidualSumOfSquaresCalculated = false;
    kIsRSquaredCalculated = false;
    kIsAdjustedRSquaredCalculated = false;
}

LinearRegressionModel::~LinearRegressionModel() = default;

void LinearRegressionModel::SetFeatureMatrix(Double_t* externalFeatureMatrix) {
    kFeatureMatrix = externalFeatureMatrix;
}

void LinearRegressionModel::SetTarget(Double_t *externalTarget) {
    kTarget = externalTarget;
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
        cost += TMath::Power(PredictTrainTarget(weights, &kFeatureMatrix[i * kNumberOfFeatures]) - kTarget[i], 2);
    }

    return cost / (2 * (Double_t)kNumberOfTrainingSamples);
}

void LinearRegressionModel::Train() {
    TMinuit* fMinimizer = new TMinuit(kNumberOfFeatures);
    fMinimizer->SetObjectFit(this);
    fMinimizer->SetFCN(fcn);
    fMinimizer->SetPrintLevel(-1);

    Int_t ierflg = 0;
    for (int i = 0; i < kNumberOfFeatures; i++) {
        std::string parName = "weight_" + std::to_string(i);
        fMinimizer->mnparm(i, parName.c_str(), 0, 0.01, 0, 0, ierflg);
    }

    // Start minimizing cost function
    fMinimizer->SetErrorDef(0.1);
    fMinimizer->SetMaxIterations(500);
    fMinimizer->Migrad();

    // Get weights
    kWeight = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));
    kWeightError = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));
    for (int i = 0; i < kNumberOfFeatures; i++) {
        Double_t paramValue = 0;
        Double_t paramError = 0;
        fMinimizer->GetParameter(i, paramValue, paramError);
        kWeight[i] = paramValue;
        kWeightError[i] = paramError;
    }

    CalculateRSquared();
    CalculateAdjustedRSquared();

    kIsTrained = true;
}

void LinearRegressionModel::SetNumberOfTrainingSamples(unsigned int n) {
    kNumberOfTrainingSamples = n;
    kFeatureMatrix = (Double_t*) malloc(kNumberOfTrainingSamples * kNumberOfFeatures * sizeof(Double_t));
}

void LinearRegressionModel::SetNumberOfFeatures(unsigned int n) {
    kNumberOfFeatures = n;
    kTarget = (Double_t*) malloc(kNumberOfTrainingSamples * 1 * sizeof(Double_t));
}

Double_t LinearRegressionModel::GetWeight(unsigned int idx) {
    return kWeight[idx];
}

Double_t LinearRegressionModel::GetWeightError(unsigned int idx) {
    return kWeightError[idx];
}

Double_t LinearRegressionModel::TrainedPredictor(Double_t *features) {
    Double_t predictedTarget = 0;
    for (int i = 0; i < kNumberOfFeatures; i++) {
        predictedTarget += features[i] * kWeight[i];
    }

    return predictedTarget;
}

void LinearRegressionModel::CalculateMeanTarget() {
    kTargetMean = 0;
    for (int i = 0; i < kNumberOfTrainingSamples; i++) {
        kTargetMean += kTarget[i];
    }
    kTargetMean = kTargetMean / kNumberOfTrainingSamples;
    kIsTargetMeanCalculated = true;
}

void LinearRegressionModel::CalculateTotalSumOfSquares() {
    if (!kIsTargetMeanCalculated) {
        CalculateMeanTarget();
    }
    kTotalSumOfSquares = 0;
    for (int i = 0; i < kNumberOfTrainingSamples; i++) {
        kTotalSumOfSquares += TMath::Power(kTarget[i] - kTargetMean, 2);
    }
    kIsTotalSumOfSquaresCalculated = true;
}

void LinearRegressionModel::CalculateResidualSumOfSquares() {
    if (!kIsTargetMeanCalculated) {
        CalculateMeanTarget();
    }

    kResidualSumOfSquares = 0;
    for (int i = 0; i < kNumberOfTrainingSamples; i++) {
        kResidualSumOfSquares += TMath::Power(kTarget[i] - TrainedPredictor(&kFeatureMatrix[i * kNumberOfFeatures]), 2);
    }
    kIsResidualSumOfSquaresCalculated = true;
}

void LinearRegressionModel::CalculateRSquared() {
    if (!kIsResidualSumOfSquaresCalculated) {
        CalculateResidualSumOfSquares();
    }

    if (!kIsTotalSumOfSquaresCalculated) {
        CalculateTotalSumOfSquares();
    }

    kRSquared = 1 - kResidualSumOfSquares / kTotalSumOfSquares;
    kIsRSquaredCalculated = true;
}

void LinearRegressionModel::CalculateAdjustedRSquared() {
    if (!kIsRSquaredCalculated) {
        CalculateRSquared();
    }
    kAdjustedRSquared = 1 - (1 - kRSquared) * (((double)kNumberOfTrainingSamples - 1.) / ((double) kNumberOfTrainingSamples -
            (double)kNumberOfFeatures));
    kIsAdjustedRSquaredCalculated = true;
}

Double_t LinearRegressionModel::GetRSquared() {
    if (!kIsTrained) {
        Train();
    }
    return kRSquared;
}

Double_t LinearRegressionModel::GetAdjustedRSquared() {
    if (!kIsTrained) {
        Train();
    }
    return kAdjustedRSquared;
}

void fcn(Int_t&, Double_t*, Double_t& cost, Double_t* weights, Int_t) {
    auto lrm = (LinearRegressionModel*) gMinuit->GetObjectFit();
    cost = lrm->CostFunction(weights);
}
