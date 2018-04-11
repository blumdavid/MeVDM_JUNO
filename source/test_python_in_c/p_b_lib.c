/* p_b_lib.c */

#include <math.h>

double f(int n, double *x, void *user_data){
	// x is an array containing the point the function f is evaluated at 
	// x[0] = number of DSNB background events, x[1] = number of CCatmo background events, x[2] = number of reactor
    // background events
		
	// user_data to arbitrary additional data you want to provide. 
    // 'void *user_data' declares a pointer, but without specifying which data type it is pointing to.
    double *p = (double *)user_data;
    double fraction = *p;
    
    //double fraction = *(double *)user_data;
    
	// get number of columns of fraction: 
    int col=sizeof(fraction)/sizeof(double);
	
	// int col = 10;
	
	/* preallocate the fractions arrays: */
	double f_dsnb[col];
	double f_ccatmo[col];
	double f_reactor[col];
	double data[col];
	
	/* fill arrays with the data */
	int l;
	for(l=0; l<col; l++)
	{
		f_dsnb[l] = fraction[l];
        f_ccatmo[l] = fraction[l + col];
        f_reactor[l] = fraction[l + 2*col];
        data[l] = fraction[l + 3*col];
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
		double lambda = x[0]*f_dsnb[j] + x[1]*f_ccatmo[j] + x[2]*f_reactor[j];
		
		/* calculate the factorial of the value of data in bin j: */
		if(data[j]>1.5){
			/* for data[j]=2,3,4,.... */
			for(i=1; i<data[j]; i++)
			{
				fakultaet = fakultaet * i;
			}
		}			
		
		/* Calculate the value of the function for the j-th bin: */
		array_1[j] = power(lambda, data[j])*exp(-lambda)/fakultaet;
	}
	
	/* calculate the product of the array array_1: */
	double prod = 1;
	for(k=0; k<col; k++)
	{
		prod = prod*array_1[k];
	}
		
	return prod;
}
