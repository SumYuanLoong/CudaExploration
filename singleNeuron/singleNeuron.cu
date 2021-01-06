#include <iostream>
#include <math.h>
#include <time.h>
#include <conio.h>
#include "cuda_runtime.h"               //headers for the cuda methods
#include "device_launch_parameters.h"
#include "Functions.cuh"

#define TMAE 0.15
#define trainspeed 0.05
#define totalRows 100
#define trRow 90 //number of rows in the training set
#define tsRow 10 //number of rows in the test set
#define col 10 //columns of data including desired value "result"

int trainingDataCount = trRow * (col - 1);
int testDataCount = tsRow * (col - 1);

//datasets
float TrainSetData[trRow][col - 1]; 
float TestSetData[tsRow][col - 1]; 
float TrainSetDiag[trRow]; //training result set 
float TestSetDiag[tsRow]; //testing result set
double trainz[trRow]; //store training set z value of each patient
double testz[tsRow]; //store testing set z value of each patient
double trainsig[trRow]; //store training set sigmoid y cap of each patient
double testsig[tsRow]; //store testing set sigmoid y cap of each patient

//pointers to the datasets

//training data set
//data is r1c1 r1c2 r1c9 r2c1 r2c2
//1 row is 1 patient
float* pTrainSetData;
//testing data set
//data is r1c1 r1c2 r1c9 r2c1 r2c2
//1 row is 1 patient
float* pTestSetData;
float* pTrainSetDiag;
float* pTestSetDiag;
double* ptrainz;
double* ptestz;

//original data/weights/bias for printing at end
double weight[9];
double bias;
double utrmmse, utsmmse, ttrmmse, ttsmmse;
double* putrmmse = &utrmmse;
double* putsmmse = &utsmmse;
double* pttrmmse = &ttrmmse;
double* pttsmmse = &ttsmmse;

void readFile(float* traindata, float* testdata, float* trainDiag, float* testDiag);
double random();
void matrix();

int main(void) {
    clock_t tstart = clock(); //start clock
    srand(time(NULL));
    
    //cuda memory allocation
    cudaMallocManaged(&pTrainSetData, trRow * (col - 1) * sizeof(float));
    cudaMallocManaged(&pTestSetData, tsRow * (col - 1) * sizeof(float));
    cudaMallocManaged(&pTrainSetDiag, trRow  * sizeof(float));
    cudaMallocManaged(&pTestSetDiag, tsRow  * sizeof(float));
    cudaMallocManaged(&ptrainz, trRow * sizeof(double));
    cudaMallocManaged(&ptestz, trRow * sizeof(double));

    readFile(pTrainSetData, pTestSetData, pTrainSetDiag, pTestSetDiag);

    
    int numBlocks = (trainingDataCount + 256 - 1) / 256;
    memset(ptrainz, 0, trRow * sizeof(double));  // set the z arr to 0 so the threads can assign values
    linearRegress <<<numBlocks, 256 >>> (trRow, pTrainSetData, ptrainz, col);
    
}

void readFile(float *traindata, float *testdata, float *trainDiag, float *testDiag) {
    int x, y;
    int a=0, b=0, c=0, d=0;
    FILE* fertfile_ptr = fopen("fertility_Diagnosis_Data_Group1_4.txt", "r");

    // error handling
    if (fertfile_ptr == NULL)
    {
        fprintf(stderr, "Error opening file: ");
        exit(EXIT_FAILURE);
    }

    for (x = 0; x < totalRows; x++) {
        for (y = 0; y < col; y++) {
            if (y == (col - 1)) { //result of diagnosis
                if (x < trRow) {
                    fscanf(fertfile_ptr, "%f, ", trainDiag);
                    trainDiag++,a++;
                }
                else {
                    fscanf(fertfile_ptr, "%f, ", testDiag);
                    testDiag++,b++;
                }
            }
            else {  //data to determine diagnosis
                if (x < trRow) {
                    fscanf(fertfile_ptr, "%f, ", traindata);
                    traindata++,c++;
                }
                else {
                    fscanf(fertfile_ptr, "%f, ", testdata);
                    testdata++,d++;
                }
            }
        }
    }
    fclose(fertfile_ptr);
    printf("%d training data read.\n", c);
    printf("%d training diag read.\n", a);

    printf("%d testing data read.\n", d);
    printf("%d testing diag read.\n", b);


}

//generate a number between -1 and 1
double random()
{
    int w;
    double resultrand;
    w = (rand() % 3) - 1; //random between int -1, 0 , 1
    if (w > 1 || w < -1)
    {
        w = (rand() % 3) - 1; //random between int -1, 0 , 1
        //printf("%d", w);
    }
    if (w == 0)
        w = 1;
    //to improve the random result for double -1.00 to 1.00 by using w
    resultrand = (1.0 * rand() / RAND_MAX - w);
    if (resultrand > 1.00)
    {
        resultrand = resultrand - 1;
    }
    //printf("\nweight = %lf", resultrand);
    return resultrand;
}

// to display the confusion matrix
void matrix() {
    int tp = 0, fp = 0, tn = 0, fn = 0, i, y;
    for (i = 0; i < trRow; i++) {
        y = round(trainsig[i]);
        if (y == 1)
        {
            if (TrainSetDiag[i] == y)
                tp++;
            else
                fp++;
        }
        else
        {
            if (TrainSetDiag[i] == y)
                tn++;
            else
                fn++;
        }
    }
    printf("\n-------------------------------------------\n\n");
    printf("Training Set Confusion Matrix\n                          True      False\n");
    printf("Predicted Positive        %d         %d\n", tp, fp);
    printf("Predicted Negative        %d        %d\n", tn, fn);
    printf("\n-------------------------------------------\n\n");
    tp = 0, fp = 0, tn = 0, fn = 0;

    for (i = 0; i < tsRow; i++) {
        y = round(testsig[i]);
        if (y == 1)
        {
            if (TestSetDiag[i] == y)
                tp++;
            else
                fp++;
        }
        else
        {
            if (TestSetDiag[i] == y)
                tn++;
            else
                fn++;
        }
    }
    printf("Testing Set Confusion Matrix\n                          True      False\n");
    printf("Predicted Positive        %d         %d\n", tp, fp);
    printf("Predicted Negative        %d         %d", tn, fn);
    printf("\n\n-------------------------------------------\n\n");
}