//
// Created by Dung Phan on 10/6/17.
//

#include "LinearRegressionData.h"

LinearRegressionData::LinearRegressionData() {
    kTestRatio = 0.2;
    kFeatureScaling = false;
    kIsMalloced = false;
}

LinearRegressionData::~LinearRegressionData() = default;

void LinearRegressionData::SetNumberOfFeatures(unsigned int n) {
    kNumberOfFeatures = n;
}

void LinearRegressionData::SetNumberOfSamples(unsigned int n) {
    kNumberOfSamples = n;
}

void LinearRegressionData::SetTestSampleRatio(double testRatio) {
    kTestRatio = testRatio;
    kNumberOfTestSamples = (unsigned int)(kTestRatio * kNumberOfSamples);
    kNumberOfTrainSamples = kNumberOfSamples - kNumberOfTestSamples;
}

void LinearRegressionData::SetFeatureScaling(bool doScalingFeatures) {
    kFeatureScaling = doScalingFeatures;
}

void LinearRegressionData::ImportFeature(Double_t *featureColumn, unsigned int featureIdx) {
    if (!kIsMalloced) {
        MallocStorage();
    }

    if (kFeatureScaling && (featureIdx != 0)) {
        kFeatureMeans[featureIdx] = GetMeanOfFeature(featureColumn);
        kFeatureStdDev[featureIdx] = GetStdDevOfFeature(featureColumn);
        ScaleFeature(featureColumn, featureIdx);
    }

    for (int i = 0; i < kNumberOfSamples; i++) {
        kFeatureMatrix[i * kNumberOfFeatures + featureIdx] = featureColumn[i];
    }
}

void LinearRegressionData::ImportTarget(Double_t *targetColumn) {
    if (!kIsMalloced) {
        MallocStorage();
    }

    for (int i = 0; i < kNumberOfSamples; i++) {
        kTarget[i] = targetColumn[i];
    }
}

void LinearRegressionData::MallocStorage() {
    kFeatureMatrix          = (Double_t*)malloc(kNumberOfSamples * kNumberOfFeatures * sizeof(Double_t));
    kTarget                 = (Double_t*)malloc(kNumberOfSamples * sizeof(Double_t));

    kTrainFeatureMatrix     = (Double_t*)malloc(kNumberOfTrainSamples * kNumberOfFeatures * sizeof(Double_t));
    kTrainTarget            = (Double_t*)malloc(kNumberOfTrainSamples * sizeof(Double_t));

    kTestFeatureMatrix      = (Double_t*)malloc(kNumberOfTestSamples * kNumberOfFeatures * sizeof(Double_t));
    kTestTarget             = (Double_t*)malloc(kNumberOfTestSamples * sizeof(Double_t));

    kTrainPredictedTarget   = (Double_t*)malloc(kNumberOfTrainSamples * sizeof(Double_t));
    kTestPredictedTarget    = (Double_t*)malloc(kNumberOfTestSamples * sizeof(Double_t));

    kFeatureMeans           = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));
    kFeatureStdDev          = (Double_t*)malloc(kNumberOfFeatures * sizeof(Double_t));

    kTrainFeatureMatrixTranspose = (Double_t*)malloc(kNumberOfFeatures * kNumberOfTrainSamples * sizeof(Double_t));
    kTestFeatureMatrixTranspose  = (Double_t*)malloc(kNumberOfFeatures * kNumberOfTestSamples * sizeof(Double_t));

    kIsMalloced = true;
}

void LinearRegressionData::ScaleFeature(Double_t *featureColumn, unsigned int featureIdx) {
    for (int i = 0; i < kNumberOfSamples; i++) {
        featureColumn[i] = (featureColumn[i] - kFeatureMeans[featureIdx]) / kFeatureStdDev[featureIdx];
    }
}

Double_t LinearRegressionData::GetMeanOfFeature(Double_t *featureColumn) {
    Double_t mean = 0;
    for (int i = 0; i < kNumberOfSamples; i++) {
        mean += featureColumn[i];
    }
    return mean / (Double_t) kNumberOfSamples;
}

Double_t LinearRegressionData::GetStdDevOfFeature(Double_t *featureColumn) {
    Double_t stddev = 0;
    Double_t mean = GetMeanOfFeature(featureColumn);
    for(int i = 0; i < kNumberOfSamples; i++) {
        stddev += TMath::Power(featureColumn[i] - mean, 2);
    }

    return TMath::Sqrt(stddev / kNumberOfSamples);
}

bool LinearRegressionData::IsTakenIntoTestSample(unsigned int chooseN, unsigned int amongAll) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double randomNumber = dis(gen);
    if (randomNumber <= ((double)chooseN / (double)amongAll)) {
        return true;
    } else {
        return false;
    }
}

void LinearRegressionData::DivideTrainTest() {
    unsigned int countTestSample = 0;
    unsigned int countTrainSample = 0;
    for (int i = 0; i < kNumberOfSamples; i++) {
        unsigned int chooseN = kNumberOfTestSamples - countTestSample;
        unsigned int amongAll = kNumberOfSamples - i;
        if (IsTakenIntoTestSample(chooseN, amongAll) && (countTestSample < kNumberOfTestSamples)) {
            for (int j = 0; j < kNumberOfFeatures; j++) {
                kTestFeatureMatrix[countTestSample * kNumberOfFeatures + j] = kFeatureMatrix[i * kNumberOfFeatures + j];
            }
            kTestTarget[countTestSample] = kTarget[i];
            countTestSample++;
        } else {
            for (int j = 0; j < kNumberOfFeatures; j++) {
                kTrainFeatureMatrix[countTrainSample * kNumberOfFeatures + j] = kFeatureMatrix[i * kNumberOfFeatures +
                                                                                               j];
            }
            kTrainTarget[countTrainSample] = kTarget[i];
            countTrainSample++;
        }
    }

    for (int i = 0; i < kNumberOfFeatures; i++) {
        for (int j = 0; j < kNumberOfTrainSamples; j++) {
            kTrainFeatureMatrixTranspose[i * kNumberOfTrainSamples + j] = kTrainFeatureMatrix[j * kNumberOfFeatures + i];
        }
    }

    for (int i = 0; i < kNumberOfFeatures; i++) {
        for (int j = 0; j < kNumberOfTestSamples; j++) {
            kTestFeatureMatrixTranspose[i * kNumberOfTestSamples + j] = kTestFeatureMatrix[j * kNumberOfFeatures + i];
        }
    }

    if (kFeatureScaling) {
        for (int i = 1; i < kNumberOfFeatures; i++) {
            for (int j = 0; j < kNumberOfTrainSamples; j++) {
                kTrainFeatureMatrixTranspose[i * kNumberOfTrainSamples + j] = ReScale(kTrainFeatureMatrixTranspose[i * kNumberOfTrainSamples + j], i);
            }
        }

        for (int i = 1; i < kNumberOfFeatures; i++) {
            for (int j = 0; j < kNumberOfTestSamples; j++) {
                kTestFeatureMatrixTranspose[i * kNumberOfTestSamples + j] = ReScale(kTestFeatureMatrixTranspose[i * kNumberOfTestSamples + j], i);
            }
        }
    }
}

Double_t *LinearRegressionData::GetTrainPredictedTarget() {
    return kTrainPredictedTarget;
}

Double_t *LinearRegressionData::GetTestPredictedTarget() {
    return kTestPredictedTarget;
}

Double_t *LinearRegressionData::GetTrainTarget() {
    return kTrainTarget;
}

Double_t *LinearRegressionData::GetTestTarget() {
    return kTestTarget;
}

Double_t *LinearRegressionData::GetTarget() {
    return kTarget;
}

Double_t *LinearRegressionData::GetFeatureMatrix() {
    return kFeatureMatrix;
}

Double_t *LinearRegressionData::GetTrainFeatureMatrix() {
    return kTrainFeatureMatrix;
}

Double_t *LinearRegressionData::GetTestFeatureMatrix() {
    return kTestFeatureMatrix;
}

unsigned int LinearRegressionData::GetNumberOfTestSamples() {
    return kNumberOfTestSamples;
}

unsigned int LinearRegressionData::GetNumberOfTrainSamples() {
    return kNumberOfTrainSamples;
}

unsigned int LinearRegressionData::GetNumberOfSamples() {
    return kNumberOfSamples;
}

unsigned int LinearRegressionData::GetNumberOfFeatures() {
    return kNumberOfFeatures;
}

void LinearRegressionData::FeatureScaleThisSample(Double_t *sampleFeatures) {
    for (int i = 1; i < kNumberOfFeatures; i++) {
        sampleFeatures[i] = (sampleFeatures[i] - kFeatureMeans[i]) / kFeatureStdDev[i];
    }
}

Double_t *LinearRegressionData::GetTrainFeature(unsigned int colIdx) {
    return &kTrainFeatureMatrixTranspose[colIdx * kNumberOfTrainSamples];
}

Double_t *LinearRegressionData::GetTestFeature(unsigned int colIdx) {
    return &kTestFeatureMatrixTranspose[colIdx * kNumberOfTestSamples];
}

Double_t LinearRegressionData::ReScale(Double_t feature, unsigned int featureIdx) {
    if (featureIdx == 0) {
        return feature;
    }

    return (feature * kFeatureStdDev[featureIdx]) + kFeatureMeans[featureIdx];
}


