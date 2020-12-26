#include "Functions.cu"

void linearRegress(short flag);
void sigmoid(double zArr[], double sigArr[], int arrSz);
void mmseFunc(double* trainMmse, double* testMmse);
double maeFunc();
void backPropagate();
double random();
void matrix();