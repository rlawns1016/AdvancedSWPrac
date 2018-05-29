#include <stdio.h>
#include <random>
#include <time.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#define DEGREE 10

#define N_X 1048576
void Eval_Poly_Naive(double y[], double x[], int n_x, double a[], int deg);
void Eval_Poly_Horner(double y[], double x[], int n_x, double a[], int deg);
__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;
void main(void)
{
	double a[DEGREE+1]; //랜덤하게 초기화
	double *x = (double *)malloc(sizeof(double)*N_X);
	double *y_n = (double *)malloc(sizeof(double)*N_X);
	double *y_h = (double *)malloc(sizeof(double)*N_X);
	int i;

	srand((unsigned)time(NULL));
	for(i = 0; i < DEGREE + 1; i++)
		a[i] = (double)rand() / ((double)RAND_MAX);
	for (i = 0; i < N_X ; i++){
		x[i] = (double)rand() / ((double)RAND_MAX);
	}
	
	//x에 100만개정도 넣어서 연산 시간 측정
	CHECK_TIME_START;
	Eval_Poly_Naive(y_n, x, N_X, a, DEGREE+1);
	CHECK_TIME_END(compute_time);
	printf("Eval_Poly_Naive: %f ms\n",compute_time);

	CHECK_TIME_START;
	Eval_Poly_Horner(y_h, x, N_X, a, DEGREE+1);
	CHECK_TIME_END(compute_time);
	printf("Eval_Poly_Horner: %f ms\n",compute_time);


	//y_n과 y_h가 비슷함을 보여야함
	
	for(i = 0; i < 10; i++)
		printf("y_n[%d] = %lf \t y_h[%d] = %lf \t (y_n - y_h)^2 = %lf \n", i, y_n[i], i, y_h[i], pow(y_n[i] - y_h[i],2));

	free(x);     
	free(y_n);
	free(y_h);
	getchar();
	/*
		horner's rule에 의한 계산의 속도가 훨씬 빠른것을 확인할 수 있었는데 
		floating point의 곱셈 연산의 횟수가 줄어들기 때문인것 같다.
	*/
}
void Eval_Poly_Naive(double y[], double x[], int n_x, double a[], int deg)
{
	//pow사용
	int i,j;
	for(i = 0; i < n_x; i++){
		y[i] = 0;
		for(j = 0; j <= deg; j++){
			y[i] += a[j] * pow(x[i] , j);
		}
	}
}
void Eval_Poly_Horner(double y[], double x[], int n_x, double a[], int deg)
{
	int i,j;
	for(i = 0; i < n_x; i++){
		y[i] = 0;
		for(j = deg; j >= 0; j--){    
			y[i] *= x[i];
			y[i] += a[j];
		}
	}
}
