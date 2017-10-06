#include <iostream>
#include <LinearRegressionModel.h>

int main() {
    unsigned int nOfFeatures = 1 + 1;
    unsigned int nOfTrainingSamples = 10;
    Double_t* FeatureMatrix = (Double_t*) malloc(nOfTrainingSamples * nOfFeatures * sizeof(Double_t));
    Double_t* TargetVector  = (Double_t*) malloc(nOfTrainingSamples * 1 * sizeof(Double_t));

    for (int i = 0; i < nOfTrainingSamples; i++) {
        FeatureMatrix[i * nOfFeatures + 0] = 1;
        FeatureMatrix[i * nOfFeatures + 1] = (Double_t)i;
        TargetVector[i] = (Double_t)i * 3 + 8;
    }

    LinearRegressionModel* lrm = new LinearRegressionModel();
    lrm->SetNumberOfFeatures(nOfFeatures);
    lrm->SetNumberOfTrainingSamples(nOfTrainingSamples);
    lrm->SetFeatureMatrix(FeatureMatrix);
    lrm->SetTargetVector(TargetVector);
    lrm->Train();

    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << lrm->GetWeight(i) << "." << std::endl;
    }

/*
 *  Testing cost function:
    Double_t* weight  = (Double_t*) malloc(nOfFeatures * sizeof(Double_t));
    weight[0] = 0;
    weight[1] = 1;
    std::cout << "Cost function : " << lrm->CostFunction(weight) << "." << std::endl;
*/
    return 0;
}