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

//datasets
float TrainSetData[trRow][col - 1]; //training data set
float TestSetData[tsRow][col - 1]; //testing data set
float TrainSetDiag[trRow]; //training result set 
float TestSetDiag[tsRow]; //testing result set
double trainz[trRow]; //store training set z value of each patient
double testz[tsRow]; //store testing set z value of each patient
double trainsig[trRow]; //store training set sigmoid y cap of each patient
double testsig[tsRow]; //store testing set sigmoid y cap of each patient

//pointers to the datasets
float* pTrainSetData = &TrainSetData[0][0];
float* pTestSetData = &TestSetData[0][0];
float* pTrainSetDiag = &TrainSetDiag[0];
float* pTestSetDiag = &TestSetDiag[0];
double* ptrainz = &trainz[0];
double* ptestz = &testz[0];

//original data/weights/bias for printing at end
double weight[9];
double bias;
double utrmmse, utsmmse, ttrmmse, ttsmmse;
double* putrmmse = &utrmmse;
double* putsmmse = &utsmmse;
double* pttrmmse = &ttrmmse;
double* pttsmmse = &ttsmmse;

void readFile();

int main(void) {
    clock_t tstart = clock(); //start clock
    srand(time(NULL));
    //everything in between
    readFile();

    
}

void readFile() {
    int x, y;
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
                    fscanf(fertfile_ptr, "%f, ", &TrainSetDiag[x]);
                }
                else {
                    fscanf(fertfile_ptr, "%f, ", &TestSetDiag[x - trRow]);
                }
            }
            else {  //data to determine diagnosis
                if (x < trRow) {
                    fscanf(fertfile_ptr, "%f, ", &TrainSetData[x][y]);
                }
                else {
                    fscanf(fertfile_ptr, "%f, ", &TestSetData[x - trRow][y]);
                }
            }
        }
    }
    fclose(fertfile_ptr);
}