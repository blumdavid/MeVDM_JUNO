/* p_sb_lib.c */

#include <math.h>

double f(int n, double *x, void *user_data) {
	/* x is an array containing the point the function f is evaluated at */
	/* x[0] = DSNB background, x[1] = CCatmo background, x[2] = reactor background, x[3] = signal */
		
	/* user_data to arbitrary additional data you want to provide. */
	double fraction = *(double *)user_data;
	/* get number of rows of fraction -> should be 5 */
	int row=(sizeof(fraction)/sizeof(fraction[0]));
	/* get number of columns of fraction -> should be , if energy array is from 15 to 25 with interval 0.1 MeV */
    int col=(sizeof(fraction)/sizeof(fraction[0][0]))/row;
	
	/* preallocate the fractions */
	double f_dsnb[col];
	double f_ccatmo[col];
	double f_reactor[col];
	double f_signal[col];
	double data[col];
	
	/* fill arrays with the data */
	int i_dsnb;
	for(i_dsnb=0; i_dsnb<col; i_dsnb++)
	{
		f_dsnb[i_dsnb] = fraction[0][i_dsnb];
	}
	
	int i_ccatmo;
	for(i_ccatmo=0; i_ccatmo<col; i_ccatmo++)
	{
		f_ccatmo[i_ccatmo] = fraction[1][i_ccatmo];
	}
	
	int i_reactor;
	for(i_reactor=0; i_reactor<col; i_reactor++)
	{
		f_reactor[i_reactor] = fraction[2][i_reactor];
	}
	
	int i_signal;
	for(i_signal=0; i_signal<col; i_signal++)
	{
		f_signal[i_signal] = fraction[3][i_signal];
	}
	
	int i_data;
	for(i_data=0; i_data<col; i_data++)
	{
		data[i_data] = fraction[4][i_data];
	}
	
	/* preallocate the array, which represents lambda_i^n_i / factorial(n_i) * exp(-lambda_i)	 */
	double array_1[col];
	int j;
	double fakultaet = 1.0;
	int i;
	int k;
		
	for(j=0; j<col; j++)
	{
		/* calculate lambda for the bin j: */
		double lambda = x[3]*f_signal[j] + x[0]*f_dsnb[j] + x[1]*f_ccatmo[j] + x[2]*f_reactor[j];
		
		/* calculate the factorial of the value of data in bin j: */
		if(data[j]>1.5){
			/* for data[j]=2,3,4,.... */
			for(i=1; i<data[j]; i++)
			{
				fakultaet = fakultaet * i;
			}
		}			
		
		/* Calculate the value of the function for the j-th bin: */
		array_1[j] = power(lambda, data[j])*exp(-lambda)/fakultaet
	}
	
	/* calculate the product of the array array_1: */
	double prod = 1;
	for(k=0; k<col; k++)
	{
		prod = prod*array_1[k];
	}
		
	return prod
}
