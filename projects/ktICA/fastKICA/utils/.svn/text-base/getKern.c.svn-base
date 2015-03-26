#include "mex.h"
#include "math.h"

/* 
   K = getKern(x,y,sigma)
   returns kernel on row vectors x and y (Gaussian),
   length(x) x length(y)

 Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton 

*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, 
		 const mxArray *prhs[]){

  double *x, *y, *k;
  int nx, ny, i, j;
  double sigma;
  
  x = mxGetPr(prhs[0]);
  y = mxGetPr(prhs[1]);
  sigma = (double) *mxGetPr(prhs[2]);
  nx = mxGetN(prhs[0]);
  ny = mxGetN(prhs[1]);
  sigma *= 2 * sigma;
  
  plhs[0] = mxCreateDoubleMatrix(nx, ny, mxREAL);
  k = mxGetPr(plhs[0]);
  
  for (i=0; i<ny; ++i){
    for (j=0; j<nx; ++j){
      k[i*nx + j] = exp(-(x[j] - y[i]) * (x[j] - y[i]) / sigma);
    }
  }
    
  return;
}
