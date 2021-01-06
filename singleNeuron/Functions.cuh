#include "Functions.cu"

/*
var1: pointer to dataset to use
var2: pointer to z set to save too 
var3: pass in column definition
Before calling method ensure that z set is cleared
*/
void linearRegress(int maxrows, float* pdataset, double* pzArr,int col);
void sigmoid(double zArr[], double sigArr[], int arrSz);
void mmseFunc(double* trainMmse, double* testMmse);
double maeFunc(int trRow, float* sigma, float* diag);
void backPropagate();