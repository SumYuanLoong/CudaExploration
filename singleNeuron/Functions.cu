__global__ void linearRegress(int maxrows, float* pdataset, double* pzArr, int col) {
    //maxRows = sizeof(TrainSetData) / sizeof(*pTrainSetData);

    int index = blockIdx.x * blockDim.x + threadIdx.x;  //block index * threads per block * thread index
    //int increment = blockDim.x * gridDim.x;             //threads per block * total threads active

    int column = index % 9;     //determine what weight to apply 
    int row = index / 9;        //to determine where to save z to



    //int a, b, c = 0; // a is loop counter, b for position in row, c for row
    //double z = 0;
    //for (a = 0, b = 0; a < maxrows; a++, pdataset++) {
    //    if (b == (col - 2)) {
    //        z += (weight[b] * *pdataset) + bias;
    //        *pzArr = z;
    //        pzArr++;   // increment to next value in z arrary
    //        b = 0;
    //        z = 0;        //reset column and z row values
    //    }
    //    else
    //    {
    //        z += (weight[b++] * *pdataset);     //calculation
    //    }
    //}
}
//
////sigmoid function taking in the z arr
//void sigmoid(double zArr[], double sigArr[], int arrSz) {
//    int i;
//    for (i = 0; i < arrSz; i++) {
//        sigArr[i] = (1 / (1 + exp(-zArr[i])));
//    }
//}
//
////computing the mmse function for the testing and training sets
//void mmseFunc(double* trainMmse, double* testMmse) {
//    int i = 0;
//    double mmsesum = 0;
//
//    for (i = 0; i < trRow; i++) {
//        mmsesum += (pow((trainsig[i] - TrainSetDiag[i]), 2));
//    }
//    *trainMmse = mmsesum / trRow;
//    mmsesum = 0;
//    for (i = 0; i < tsRow; i++) {
//        mmsesum += (pow((testsig[i] - TestSetDiag[i]), 2));
//    }
//    *testMmse = mmsesum / tsRow;
//}

//calculation of mae for training set mae is only dependent on training set
double maeFunc(int trRow, float *sigma, float *diag) {
    int i;
    double maesum = 0;
    for (i = 0; i < trRow; i++) {
        maesum += fabs(*sigma - *diag);
    }
    return maesum / 90;
}

////backPropagate
//void backPropagate() {
//    int x, y;
//    double sumtrainw = 0, sumtrainb = 0;
//    for (y = 0; y < col - 1; y++)
//    {
//        //printf("\nsummation z[%d] = %f", a, trainz[a]);
//        //printf("\nsigmoid y[%d] = %f d[%d] = %f", a, trainsig[a], a, trainoutpdata[a][0]);
//        //printf("\nuntrained mmse (1*(summation ycap - d)^2)/90 = %f", ummse);
//        //printf("\nmae (1*(summation ycap - d))/90 = %f\n", mae);
//
//        for (x = 0; x < trRow; x++)
//        {
//            sumtrainw += (trainsig[x] - TrainSetDiag[x]) * (exp(trainz[x]) / ((1 + exp(trainz[x])) * (1 + exp(trainz[x]))) * TrainSetData[x][y]);
//            //printf("\nsumtrainw[%d][%d] = %f", a, b, sumtrainw);
//            if (y == 8)
//            {
//                sumtrainb += (trainsig[x] - TrainSetDiag[x]) * (exp(trainz[x]) / ((1 + exp(trainz[x])) * (1 + exp(trainz[x]))) * 1);
//                //printf("\nsumtrainb[%d][%d] = %f", x, y, sumtrainb);
//
//            }
//        }
//        sumtrainw = (sumtrainw / trRow);
//        weight[y] = (weight[y] - (trainspeed * sumtrainw)); //update the new weight into oldw[0-8]
//        //printf("\ntrainedw[%d][%d] = %f", x, y, weight[y]);
//        sumtrainb = (sumtrainb / trRow);
//        bias = (bias - (trainspeed * sumtrainb)); //update the new bias b into oldb
//        sumtrainw = 0;
//        sumtrainb = 0;
//    }
//}