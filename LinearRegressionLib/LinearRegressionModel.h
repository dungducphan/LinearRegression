//
// Created by Dung Phan on 10/5/17.
//

#ifndef LINEARREGRESSION_LINEARREGRESSIONMODEL_H
#define LINEARREGRESSION_LINEARREGRESSIONMODEL_H

#include <TMinuit.h>

class LinearRegressionModel {
public:
    LinearRegressionModel();
    virtual ~LinearRegressionModel();

protected:
    double LinearRegressor();
    double CostFuntion();
    double Train();
    double Predict();

private:
    TMinuit*     fMinimizer;

    unsigned int kNumberOfFeatures;
    unsigned int kNumberOfTrainingSamples;
    double*      kWeights;
};


#endif //LINEARREGRESSION_LINEARREGRESSIONMODEL_H
