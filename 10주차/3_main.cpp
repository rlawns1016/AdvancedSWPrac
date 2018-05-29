#include <stdio.h>
#include <random>
#include <time.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#define N 24
double Taylor_series(double x, int n);
double Taylor_series_ex(double x, int n);
__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;

void main(void)
{
	double x,result;

	x = -8.3; 
	result = Taylor_series(x, N);
	printf("%e\n",result);

	x = -8.3;
	result = Taylor_series_ex(x, N);
	printf("%e\n",result);

	getchar();
	/*
		(ii) ���α׷��� �������� 4.872486..�� ��Ȯ�� ���� ���� ���Ͽ���.
		�̴� ����� floating point���� ���� ������ ���� �Ͼ�� �����̴�.
		(iii) ����� floating point���� ���� ������ ���ϱ� ���Ͽ� x���� -1�� �����ְ�
		���ϰ��� ������ ���Ͽ��־���.
		�̷��� ����� ���Ͽ��� �� ��Ȯ�� ���� ���� �� �־���.
	*/
}




double Taylor_series(double x, int n)
{
	int i;
	double result = 0, temp = 1, fact = 1;

	for (i = 1; i<=n; i++)
	{
		temp *= x;
		fact *= i;     
		result += temp / fact;
	}
	return result+1;

}

double Taylor_series_ex(double x, int n)
{
	int i;
	double result=1;
	int m = n;

	if(x < 0){
		x *= -1;
		for(i = n; i > 0; i--,m--)
			result = 1 + result * (x / m);

		return 1/result;
	}
	else
	{
		for(i = n; i > 0; i--,m--)
			result = 1 + result * (x / m);
		return result;
	}
}
