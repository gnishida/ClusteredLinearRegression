#include "ClusteredLinearRegression.h"
#include "MLUtils.h"

int main() {
	cv::Mat_<double> X;
	ml::loadDataset("dataX.txt", X);
	cv::Mat_<double> Y;
	ml::loadDataset("dataY.txt", Y);

	ClusteredLinearRegression clr(X, Y, 5);
	for (int i = 0; i < X.rows; ++i) {
		cv::Mat_<double> Y_hat = clr.predict(X.row(i));
		cout << Y.row(i) << "->" << Y_hat << endl;
	}

	return 0;
}