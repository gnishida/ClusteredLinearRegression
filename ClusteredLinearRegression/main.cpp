#include "ClusteredLinearRegression.h"
#include "MLUtils.h"

using namespace std;

int main(int argc,char *argv[]) {
	if (argc < 4) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename of X> <filename of Y> <min cluster size>" << endl;
		cout << endl;

		return -1;
	}

	cv::Mat_<double> X, Y;
	ml::loadDataset(argv[1], X);
	ml::loadDataset(argv[2], Y);

	cv::Mat_<double> normalizedX, meanX, stddevX;
	ml::normalizeDataset(X, normalizedX, meanX, stddevX);
	cv::Mat_<double> normalizedY, meanY, stddevY;
	ml::normalizeDataset(Y, normalizedY, meanY, stddevY);

	cv::Mat_<double> trainingX, testX;
	ml::splitDataset(normalizedX, 0.9, trainingX, testX);
	cv::Mat_<double> trainingY, testY;
	ml::splitDataset(normalizedY, 0.9, trainingY, testY);


	cv::Mat_<double> predY(testY.rows, testY.cols);

	ClusteredLinearRegression clr(trainingX, trainingY, atoi(argv[3]));
	for (int i = 0; i < testX.rows; ++i) {
		cv::Mat_<double> normalized_y_hat = clr.predict(testX.row(i));
		normalized_y_hat.copyTo(predY.row(i));
	}

	cout << "RMSE: " << ml::rmse(testY, predY, false) << " (per cell) " << ml::rmse(testY, predY, true) << endl;

	return 0;
}