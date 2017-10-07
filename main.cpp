#include <iostream>
#include <fstream>
#include <vector>
#include <LinearRegressor.h>

#include <TGraph.h>
#include <TMultiGraph.h>
#include <TCanvas.h>
#include <TF1.h>

int main() {

/*
 * Example 1:
 * Testing LinearRegressionModel class with generated data.
 *
    unsigned int nOfFeatures = 1 + 3;
    unsigned int nOfSamples = 10;
    Double_t* FeatureMatrix = (Double_t*) malloc(nOfSamples * nOfFeatures * sizeof(Double_t));
    Double_t* TargetVector  = (Double_t*) malloc(nOfSamples * 1 * sizeof(Double_t));

    for (int i = 0; i < nOfSamples; i++) {
        FeatureMatrix[i * nOfFeatures + 0] = 1;
        FeatureMatrix[i * nOfFeatures + 1] = (Double_t)i;
        FeatureMatrix[i * nOfFeatures + 2] = (Double_t)i;
        FeatureMatrix[i * nOfFeatures + 3] = (Double_t)i;
        TargetVector[i] = (Double_t)i * 3 + 8;
    }

    LinearRegressionModel* lrm = new LinearRegressionModel();
    lrm->SetNumberOfFeatures(nOfFeatures);
    lrm->SetNumberOfTrainingSamples(nOfSamples);
    lrm->SetFeatureMatrix(FeatureMatrix);
    lrm->SetTarget(TargetVector);
    lrm->Train();

    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << lrm->GetWeight(i) << "." << std::endl;
    }
*/

/*
 * Example 2:
 * Testing LinearRegressionModel's cost function with generated data.
 *
    unsigned int nOfFeatures = 1 + 3;
    unsigned int nOfSamples = 10;
    Double_t* FeatureMatrix = (Double_t*) malloc(nOfSamples * nOfFeatures * sizeof(Double_t));
    Double_t* TargetVector  = (Double_t*) malloc(nOfSamples * 1 * sizeof(Double_t));

    for (int i = 0; i < nOfSamples; i++) {
        FeatureMatrix[i * nOfFeatures + 0] = 1;
        FeatureMatrix[i * nOfFeatures + 1] = (Double_t)i;
        FeatureMatrix[i * nOfFeatures + 2] = (Double_t)i;
        FeatureMatrix[i * nOfFeatures + 3] = (Double_t)i;
        TargetVector[i] = (Double_t)i * 3 + 8;
    }

    LinearRegressionModel* lrm = new LinearRegressionModel();
    lrm->SetNumberOfFeatures(nOfFeatures);
    lrm->SetNumberOfTrainingSamples(nOfSamples);
    lrm->SetFeatureMatrix(FeatureMatrix);
    lrm->SetTarget(TargetVector);

    Double_t* weight  = (Double_t*) malloc(nOfFeatures * sizeof(Double_t));
    weight[0] = 8;
    weight[1] = 1;
    weight[2] = 1;
    weight[3] = 1;
    std::cout << "Cost function : " << lrm->CostFunction(weight) << "." << std::endl;
*/

/*
 * Example 3:
 * Testing LinearRegressionModel class with external data.
 *
    std::vector<double> rnd, admin, pr, revenue;
    double tmp_rnd, tmp_admin, tmp_pr, tmp_revenue;
    ifstream data("test.csv");

    while (data >> tmp_rnd >> tmp_admin >> tmp_pr >> tmp_revenue) {
        rnd.push_back(tmp_rnd);
        admin.push_back(tmp_admin);
        pr.push_back(tmp_pr);
        revenue.push_back(tmp_revenue);
    }

    unsigned int numberOfRemovedFeatures = 0;

    unsigned int nOfFeatures = 1 + 3 - numberOfRemovedFeatures;
    unsigned int nOfSamples = static_cast<unsigned int>(revenue.size());
    Double_t* FeatureMatrix = (Double_t*) malloc(nOfSamples * nOfFeatures * sizeof(Double_t));
    Double_t* TargetVector  = (Double_t*) malloc(nOfSamples * 1 * sizeof(Double_t));
    for (int i = 0; i < nOfSamples; i++) {
        FeatureMatrix[i * nOfFeatures + 0] = 1;
        FeatureMatrix[i * nOfFeatures + 1] = rnd.at(i);
        FeatureMatrix[i * nOfFeatures + 2] = admin.at(i);
        FeatureMatrix[i * nOfFeatures + 3] = pr.at(i);
        TargetVector [i]                   = revenue.at(i);
    }

    LinearRegressionModel* lrm = new LinearRegressionModel();
    lrm->SetNumberOfFeatures(nOfFeatures);
    lrm->SetNumberOfTrainingSamples(nOfSamples);
    lrm->SetFeatureMatrix(FeatureMatrix);
    lrm->SetTarget(TargetVector);
    lrm->Train();
    std::cout << "Model R-Squared   : " << lrm->GetRSquared() << "." << std::endl;
    std::cout << "Adjusted R-Squared: " << lrm->GetAdjustedRSquared() << "." << std::endl;
    std::cout << std::endl;


    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << lrm->GetWeight(i) << "." << std::endl;
    }

    Double_t testFeature[4] = {1, 165349.2, 100897.8, 471784.1};
    std::cout << lrm->TrainedPredictor(testFeature) << std::endl;

    data.close();
*/

/*
 * Example 4:
 * Incorporate Data Management.
 *
    std::vector<double> rnd, admin, pr, revenue;
    double tmp_rnd, tmp_admin, tmp_pr, tmp_revenue;
    ifstream data("test.csv");

    while (data >> tmp_rnd >> tmp_admin >> tmp_pr >> tmp_revenue) {
        rnd.push_back(tmp_rnd);
        admin.push_back(tmp_admin);
        pr.push_back(tmp_pr);
        revenue.push_back(tmp_revenue);
    }

    unsigned int nOfFeatures = 1 + 3;
    unsigned int nOfSamples = static_cast<unsigned int>(revenue.size());
    Double_t* Bias      = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* RnD       = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Admin     = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* PR        = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Revenue   = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    for (int i = 0; i < nOfSamples; i++) {
        Bias[i] = 1;
        RnD[i] = rnd.at(i);
        Admin[i] = admin.at(i);
        PR[i] = pr.at(i);
        Revenue[i] = revenue.at(i);
    }

    LinearRegressionData* lrdata = new LinearRegressionData();
    lrdata->SetNumberOfFeatures(nOfFeatures);
    lrdata->SetNumberOfSamples(nOfSamples);
    lrdata->SetFeatureScaling(true);
    lrdata->SetTestSampleRatio(0.2);
    lrdata->ImportFeature(Bias, 0);
    lrdata->ImportFeature(RnD, 1);
    lrdata->ImportFeature(Admin, 2);
    lrdata->ImportFeature(PR, 3);
    lrdata->ImportTarget(Revenue);
    lrdata->DivideTrainTest();

    LinearRegressionModel* lrm = new LinearRegressionModel();
    lrm->SetNumberOfFeatures(nOfFeatures);
    lrm->SetNumberOfTrainingSamples(nOfSamples);
    lrm->SetFeatureMatrix(lrdata->GetTrainFeatureMatrix());
    lrm->SetTarget(lrdata->GetTrainTarget());
    lrm->Train();

    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << lrm->GetWeight(i) << "." << std::endl;
    }

    Double_t testFeature[4] = {1, 165349.2, 136897.8, 471784.1};
    lrdata->FeatureScaleThisSample(testFeature);
    std::cout << lrm->TrainedPredictor(testFeature) << std::endl;

    data.close();
*/

/*
 * Example 5:
 * Everything is wrapped around the LinearRegressor class now.
 */
    std::vector<double> rnd, admin, pr, revenue;
    double tmp_rnd, tmp_admin, tmp_pr, tmp_revenue;
    ifstream data("test.csv");

    while (data >> tmp_rnd >> tmp_admin >> tmp_pr >> tmp_revenue) {
        rnd.push_back(tmp_rnd);
        admin.push_back(tmp_admin);
        pr.push_back(tmp_pr);
        revenue.push_back(tmp_revenue);
    }

    unsigned int numberOfRemovedFeatures = 2;

    unsigned int nOfFeatures = 1 + 3 - numberOfRemovedFeatures;
    unsigned int nOfSamples = static_cast<unsigned int>(revenue.size());
    Double_t* Bias      = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* RnD       = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Admin     = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* PR        = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Revenue   = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    for (int i = 0; i < nOfSamples; i++) {
        Bias[i] = 1;
        RnD[i] = rnd.at(i);
        Admin[i] = admin.at(i);
        PR[i] = pr.at(i);
        Revenue[i] = revenue.at(i);
    }

    LinearRegressor* LR = new LinearRegressor();
    LR->SetNumberOfFeatures(nOfFeatures);
    LR->SetNumberOfSamples(nOfSamples);
    LR->SetFeatureScaling(false);
    LR->ImportFeature(Bias, 0);
    LR->ImportFeature(RnD, 1);
    //LR->ImportFeature(Admin, 2);
    //LR->ImportFeature(PR, 3);
    LR->ImportTarget(Revenue);
    LR->SetTestSampleRatio(0.2);
    LR->DivideTrainTest();
    LR->Train();

    std::cout << std::endl;
    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << LR->GetWeight(i) << "." << std::endl;
    }
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "R-Squared          : " << LR->GetRSquared() << "." << std::endl;
    std::cout << "Adjusted R-Squared : " << LR->GetAdjustedRSquared() << "." << std::endl;
    std::cout << std::endl;

    LR->Test();
    LR->PrintTrainTarget();
    LR->PrintTestTarget();

    unsigned int nTrain = LR->GetNumberOfTrainSamples();
    unsigned int nTest  = LR->GetNumberOfTestSamples();

    Double_t* TrainFeature = (Double_t*)malloc(nTrain * sizeof(Double_t));
    Double_t* TrainTarget  = (Double_t*)malloc(nTrain * sizeof(Double_t));
    Double_t* TrainPredictedTarget  = (Double_t*)malloc(nTrain * sizeof(Double_t));

    Double_t* TestFeature  = (Double_t*)malloc(nTest * sizeof(Double_t));
    Double_t* TestTarget  = (Double_t*)malloc(nTest * sizeof(Double_t));
    Double_t* TestPredictedTarget  = (Double_t*)malloc(nTest * sizeof(Double_t));

    for (int i = 0; i < nTrain; i++) {
        TrainFeature[i] = LR->GetTrainFeature(1)[i];
        TrainTarget[i] = LR->GetTrainTarget()[i];
        TrainPredictedTarget[i] = LR->GetTrainPredictedTarget()[i];

    }

    for (int i = 0; i < nTest; i++) {
        TestFeature[i] = LR->GetTestFeature(1)[i];
        TestTarget[i] = LR->GetTestTarget()[i];
        TestPredictedTarget[i] = LR->GetTestPredictedTarget()[i];

    }

    TGraph* trainGr = new TGraph(nTrain, TrainFeature, TrainTarget);
    trainGr->SetMarkerStyle(4);
    trainGr->SetMarkerSize(1);
    trainGr->SetMarkerColor(kRed + 1);

    TGraph* testGr  = new TGraph(nTest, TestFeature, TestTarget);
    testGr->SetMarkerStyle(4);
    testGr->SetMarkerSize(1);
    testGr->SetMarkerColor(kBlue + 2);

    TGraph* trainPredictGr = new TGraph(nTrain, TrainFeature, TrainPredictedTarget);
    trainPredictGr->SetMarkerStyle(5);
    trainPredictGr->SetMarkerSize(1);
    trainPredictGr->SetMarkerColor(kRed);

    TGraph* testPredictGr  = new TGraph(nTest, TestFeature, TestPredictedTarget);
    testPredictGr->SetMarkerStyle(5);
    testPredictGr->SetMarkerSize(1);
    testPredictGr->SetMarkerColor(kBlue);

    TMultiGraph* mg = new TMultiGraph();
    mg->Add(trainGr);
    mg->Add(testGr);
    mg->Add(trainPredictGr);
    mg->Add(testPredictGr);

    TF1* regressFunction = new TF1("regressFunction", "[0] + [1]*x", 0, 180E3);
    regressFunction->SetParameter(0, LR->GetWeight(0));
    regressFunction->SetParameter(1, LR->GetWeight(1));

    TCanvas* c = new TCanvas();
    mg->Draw("AP");
    regressFunction->Draw("same");
    c->SaveAs("test.png");
/**/

    /*
     * TODO
     *  - Example 6 returns weird value of predicted targets when
     *  using FeatureScaling. Investigate this problem.
     */

/*
 * Example 6:
 * Test with dataset with more features and samples.
 *
    std::vector<double> rnd, admin, pr, revenue, suction, level;
    double tmp_rnd, tmp_admin, tmp_pr, tmp_revenue, tmp_suction, tmp_level;
    ifstream data("test2.csv");

    while (data >> tmp_rnd >> tmp_admin >> tmp_pr >> tmp_revenue >> tmp_suction >> tmp_level) {
        rnd.push_back(tmp_rnd);
        admin.push_back(tmp_admin);
        pr.push_back(tmp_pr);
        revenue.push_back(tmp_revenue);
        suction.push_back(tmp_suction);
        level.push_back(tmp_level);
    }

    unsigned int numberOfRemovedFeatures = 0;

    unsigned int nOfFeatures = 1 + 5 - numberOfRemovedFeatures;
    unsigned int nOfSamples = static_cast<unsigned int>(revenue.size());
    Double_t* Bias      = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* RnD       = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Admin     = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* PR        = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Revenue   = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Suction   = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    Double_t* Level     = (Double_t*) malloc(nOfSamples * sizeof(Double_t));
    for (int i = 0; i < nOfSamples; i++) {
        Bias[i] = 1;
        RnD[i] = rnd.at(i);
        Admin[i] = admin.at(i);
        PR[i] = pr.at(i);
        Revenue[i] = revenue.at(i);
        Suction[i] = suction.at(i);
        Level[i] = level.at(i);
    }

    LinearRegressor* LR = new LinearRegressor();
    LR->SetNumberOfFeatures(nOfFeatures);
    LR->SetNumberOfSamples(nOfSamples);
    LR->SetFeatureScaling(true);
    LR->ImportFeature(Bias, 0);
    LR->ImportFeature(RnD, 1);
    LR->ImportFeature(Admin, 2);
    LR->ImportFeature(PR, 3);
    LR->ImportFeature(Revenue, 4);
    LR->ImportFeature(Suction, 5);
    LR->ImportTarget(Level);
    LR->SetTestSampleRatio(0.);
    LR->DivideTrainTest();
    LR->Train();

    std::cout << std::endl;
    for (int i = 0; i < nOfFeatures; i++) {
        std::cout << "Weight #" << i << ": " << LR->GetWeight(i) << "." << std::endl;
    }
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "R-Squared          : " << LR->GetRSquared() << "." << std::endl;
    std::cout << "Adjusted R-Squared : " << LR->GetAdjustedRSquared() << "." << std::endl;
    std::cout << std::endl;

    LR->Test();
    LR->PrintTrainTarget();
    LR->PrintTestTarget();


    unsigned int nTrain = LR->GetNumberOfTrainSamples();
    unsigned int nTest  = LR->GetNumberOfTestSamples();

    Double_t* TrainFeature = (Double_t*)malloc(nTrain * sizeof(Double_t));
    Double_t* TrainTarget  = (Double_t*)malloc(nTrain * sizeof(Double_t));
    Double_t* TrainPredictedTarget  = (Double_t*)malloc(nTrain * sizeof(Double_t));

    Double_t* TestFeature  = (Double_t*)malloc(nTest * sizeof(Double_t));
    Double_t* TestTarget  = (Double_t*)malloc(nTest * sizeof(Double_t));
    Double_t* TestPredictedTarget  = (Double_t*)malloc(nTest * sizeof(Double_t));

    for (int i = 0; i < nTrain; i++) {
        TrainFeature[i] = LR->GetTrainFeature(1)[i];
        TrainTarget[i] = LR->GetTrainTarget()[i];
        TrainPredictedTarget[i] = LR->GetTrainPredictedTarget()[i];

    }

    for (int i = 0; i < nTest; i++) {
        TestFeature[i] = LR->GetTestFeature(1)[i];
        TestTarget[i] = LR->GetTestTarget()[i];
        TestPredictedTarget[i] = LR->GetTestPredictedTarget()[i];

    }

    TGraph* trainGr = new TGraph(nTrain, TrainFeature, TrainTarget);
    trainGr->SetMarkerStyle(4);
    trainGr->SetMarkerSize(1);
    trainGr->SetMarkerColor(kRed + 1);

    TGraph* testGr  = new TGraph(nTest, TestFeature, TestTarget);
    testGr->SetMarkerStyle(4);
    testGr->SetMarkerSize(1);
    testGr->SetMarkerColor(kBlue + 2);

    TGraph* trainPredictGr = new TGraph(nTrain, TrainFeature, TrainPredictedTarget);
    trainPredictGr->SetMarkerStyle(5);
    trainPredictGr->SetMarkerSize(1);
    trainPredictGr->SetMarkerColor(kRed);

    TGraph* testPredictGr  = new TGraph(nTest, TestFeature, TestPredictedTarget);
    testPredictGr->SetMarkerStyle(5);
    testPredictGr->SetMarkerSize(1);
    testPredictGr->SetMarkerColor(kBlue);

    TMultiGraph* mg = new TMultiGraph();
    mg->Add(trainGr);
    mg->Add(testGr);
    mg->Add(trainPredictGr);
    mg->Add(testPredictGr);

    TF1* regressFunction = new TF1("regressFunction", "[0] + [1]*x", 0, 180E3);
    regressFunction->SetParameter(0, LR->GetWeight(0));
    regressFunction->SetParameter(1, LR->GetWeight(1));

    TCanvas* c = new TCanvas();
    mg->Draw("AP");
    regressFunction->Draw("same");
    c->SaveAs("test.png");
*/

    return 0;
}