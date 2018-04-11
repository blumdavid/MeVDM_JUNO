/* p_b_lib.c */

#include <math.h>

double f(int n, double *x, double *user_data){
	// x is an array containing the point the function f is evaluated at 
	// x[0] = number of DSNB background events, x[1] = number of CCatmo background events, x[2] = number of reactor
    // background events
    
	// user_data to arbitrary additional data you want to provide. 
    // 'void *user_data' declares a pointer, but without specifying which data type it is pointing to.
    
    double fraction = user_data[0];
    
    double c = user_data[1];
    
    double d = user_data[2];   
		
	return fraction + c + d + x[0] * x[1] * x[2];
}
