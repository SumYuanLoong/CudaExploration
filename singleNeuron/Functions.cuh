#include "Functions.cu"

void linearRegress(int maxrows, float* pdataset, double* pzArr,int col);
void sigmoid(double zArr[], double sigArr[], int arrSz);
void mmseFunc(double* trainMmse, double* testMmse);
double maeFunc(int trRow, float* sigma, float* diag);
void backPropagate();