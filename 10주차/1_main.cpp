#include <stdio.h>
#include <random>
#include <time.h>

#include <math.h>
#include <time.h>
#include <Windows.h>

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;


#define MATDIM 1024
double *pMatA, *pMatB, *pMatC, *pTransMat;
void MultiplySquareMatrices_1(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_2(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_3(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_4(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void trans_MatMat(double *pTMat, double *pMatrix, int MatSize);
void init_MatMat(void);

void main(void)
{
	init_MatMat();

	CHECK_TIME_START;
	//Sleep(500);
	CHECK_TIME_END(compute_time);
	printf("None Fun Time = %f ms\n", compute_time);
	CHECK_TIME_START;
	MultiplySquareMatrices_1(pMatA,pMatB,pMatC,MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_1 = %f ms\n", compute_time);

	for (int i = 0; i < MATDIM*MATDIM; i++)
		pMatC[i] = 0;
	CHECK_TIME_START;
	trans_MatMat(pTransMat, pMatB, MATDIM);
	MultiplySquareMatrices_2(pMatA,pTransMat,pMatC,MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_2 = %f ms\n", compute_time);

	for (int i = 0; i < MATDIM*MATDIM; i++)
		pMatC[i] = 0;
	CHECK_TIME_START;
	trans_MatMat(pTransMat, pMatB, MATDIM);
	MultiplySquareMatrices_3(pMatA,pMatB,pMatC,MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_3 = %f ms\n", compute_time);

	for (int i = 0; i < MATDIM*MATDIM; i++)
		pMatC[i] = 0;
	CHECK_TIME_START;
	trans_MatMat(pTransMat, pMatB, MATDIM);
	MultiplySquareMatrices_4(pMatA,pMatB,pMatC,MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_4 = %f ms\n", compute_time);

	getchar();

	/*
		(1) 계산속도 향상을 위하여 loop unrolling을 사용하였다.

		(2) loop를 돌때 loop count의 증가와 loop 조건을 확인하는 연산이 줄어듦으로 loop unrolling의 속도가 빠르다.

		(3) m값이 8일때 3195ms
			m값이 16일때 3116ms
			m값이 32일때 3179ms
			따라서 m의 값이 16일때 가장 효과적이었다.
	*/
}

void MultiplySquareMatrices_1(double *pDestMatrix, double *pLeftMatrix ,double *pRightMatrix, int MatSize)
{
	//손계산하는 순서대로
	int i,j,k;
	for(i=0;i<MatSize;i++){
		for(j=0;j<MatSize;j++){
			for(k=0;k<MatSize;k++){   
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k] * pRightMatrix[MatSize*k + j];
			}
		}
	}
}

void MultiplySquareMatrices_2(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	//b -> transpose
	int i,j,k;
	for(i=0;i<MatSize;i++){
		for(j=0;j<MatSize;j++){
			for(k=0;k<MatSize;k++){  
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k] * pRightMatrix[j*MatSize + k];
			}
		}	
	}
}
void MultiplySquareMatrices_3(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	//loop unrolling 8번정도
	int i,j,k;
	for(i=0;i<MatSize;i++){
		for(j=0;j<MatSize;j++){  
			for(k=0;k<MatSize/8;k++){
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8] * pRightMatrix[j*MatSize + k*8];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+1] * pRightMatrix[j*MatSize + k*8+1];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+2] * pRightMatrix[j*MatSize + k*8+2];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+3] * pRightMatrix[j*MatSize + k*8+3];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+4] * pRightMatrix[j*MatSize + k*8+4];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+5] * pRightMatrix[j*MatSize + k*8+5];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+6] * pRightMatrix[j*MatSize + k*8+6];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*8+7] * pRightMatrix[j*MatSize + k*8+7];
			}
		}	
	}
}
void MultiplySquareMatrices_4(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	//푸는 횟수를 다르게
	int i,j,k;
	
	for(i=0;i<MatSize;i++){
		for(j=0;j<MatSize;j++){
			for(k=0;k<MatSize/16;k++){     
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16] * pRightMatrix[j*MatSize + k*16];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+1] * pRightMatrix[j*MatSize + k*16+1];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+2] * pRightMatrix[j*MatSize + k*16+2];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+3] * pRightMatrix[j*MatSize + k*16+3];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+4] * pRightMatrix[j*MatSize + k*16+4];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+5] * pRightMatrix[j*MatSize + k*16+5];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+6] * pRightMatrix[j*MatSize + k*16+6];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+7] * pRightMatrix[j*MatSize + k*16+7];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+8] * pRightMatrix[j*MatSize + k*16+8];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+9] * pRightMatrix[j*MatSize + k*16+9];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+10] * pRightMatrix[j*MatSize + k*16+10];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+11] * pRightMatrix[j*MatSize + k*16+11];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+12] * pRightMatrix[j*MatSize + k*16+12];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+13] * pRightMatrix[j*MatSize + k*16+13];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+14] * pRightMatrix[j*MatSize + k*16+14];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*16+15] * pRightMatrix[j*MatSize + k*16+15];
			}
		}	
	}
	
	/*
	for(i=0;i<MatSize;i++){
		for(j=0;j<MatSize;j++){
			for(k=0;k<MatSize/32;k++){     
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32] * pRightMatrix[j*MatSize + k*32];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+1] * pRightMatrix[j*MatSize + k*32+1];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+2] * pRightMatrix[j*MatSize + k*32+2];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+3] * pRightMatrix[j*MatSize + k*32+3];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+4] * pRightMatrix[j*MatSize + k*32+4];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+5] * pRightMatrix[j*MatSize + k*32+5];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+6] * pRightMatrix[j*MatSize + k*32+6];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+7] * pRightMatrix[j*MatSize + k*32+7];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+8] * pRightMatrix[j*MatSize + k*32+8];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+9] * pRightMatrix[j*MatSize + k*32+9];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+10] * pRightMatrix[j*MatSize + k*32+10];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+11] * pRightMatrix[j*MatSize + k*32+11];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+12] * pRightMatrix[j*MatSize + k*32+12];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+13] * pRightMatrix[j*MatSize + k*32+13];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+14] * pRightMatrix[j*MatSize + k*32+14];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+15] * pRightMatrix[j*MatSize + k*32+15];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+16] * pRightMatrix[j*MatSize + k*32+16];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+17] * pRightMatrix[j*MatSize + k*32+17];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+18] * pRightMatrix[j*MatSize + k*32+18];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+19] * pRightMatrix[j*MatSize + k*32+19];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+20] * pRightMatrix[j*MatSize + k*32+20];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+21] * pRightMatrix[j*MatSize + k*32+21];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+22] * pRightMatrix[j*MatSize + k*32+22];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+23] * pRightMatrix[j*MatSize + k*32+23];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+24] * pRightMatrix[j*MatSize + k*32+24];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+25] * pRightMatrix[j*MatSize + k*32+25];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+26] * pRightMatrix[j*MatSize + k*32+26];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+27] * pRightMatrix[j*MatSize + k*32+27];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+28] * pRightMatrix[j*MatSize + k*32+28];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+29] * pRightMatrix[j*MatSize + k*32+29];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+30] * pRightMatrix[j*MatSize + k*32+30];
				pDestMatrix[i*MatSize + j] += pLeftMatrix[i*MatSize + k*32+31] * pRightMatrix[j*MatSize + k*32+31];
			}
		}
	}	
	*/
}

void trans_MatMat(double *pTMat, double *pMatrix, int MatSize)
{
	int i,j;
	for(i = 0; i < MatSize; i++){
		for(j = 0; j < MatSize; j ++){
			pTMat[i*MatSize + j] = pMatrix[j*MatSize + i];
		}
	}
}
void init_MatMat(void)
{
	double *ptr;
	pMatA = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	pMatB = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	pMatC = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	pTransMat = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	srand((unsigned)time(NULL));
	ptr = pMatA;
	for (int i = 0; i < MATDIM*MATDIM; i++)
		*ptr++ = (double)rand() / ((double)RAND_MAX);
	ptr = pMatB;
	for (int i = 0; i < MATDIM*MATDIM; i++)
		*ptr++ = (double)rand() / ((double)RAND_MAX);
}
