#include "mex.h"

/*
  dk = dChol2(HL, Ldd, Knd, Kdd, dKnd, dKdd)
 
  derivative of pairwise HSIC for kernels K and L
  HL:       column-centered n x d submatrix of L
  Ldd, Kdd: inverse d x d submatrices
  Knd:      n x d submatrix of K
  dKnd:     d(Knd) 
  dKdd:     derivative of ddimK x dddimK submatrix (both vec'ed)

  computes
  2 * vec(HL*Ldd*HL' * Knd * Kdd)' * dKnd 
  - vec(Kdd*Knd* HL*Ldd*HL' *Knd'*Kdd) * dKdd

 Copyright 2007 Stefanie Jegelka, Hao Shen, Arthur Gretton 

*/

#ifdef _WIN32
  double ddot(int*, double*, int*, double*, int*);
#else
  double ddot_(int*, double*, int*, double*, int*);
#endif

void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{ 
  /* matrices */
  double *hl, *y, *ldd, *kdd, *knd, *dkdd, *dknd, *tmp, *tmp1, *tmp3, *res;
  
  /* n: number of samples, 
     ddimK,ddimL: number of cholesky indices, 
     m: number of sources */
  int n, ddimK, ddimL, m;
  
  /* for multiplication */
  int oneI = 1, nd;
  double nullD = 0.0, oneD = 1.0, mOneD = -1.0, twoD = 2.0;
  char *chn = "N", *cht = "T";

  hl = mxGetPr(prhs[0]);
  ldd = mxGetPr(prhs[1]);
  knd = mxGetPr(prhs[2]);
  kdd = mxGetPr(prhs[3]);
  dknd = mxGetPr(prhs[4]);
  dkdd = mxGetPr(prhs[5]);

  ddimK = mxGetN(prhs[3]);
  ddimL = mxGetN(prhs[1]);
  n= mxGetM(prhs[2]);
  m = mxGetN(prhs[4]);
  nd = n*ddimK;

     
  tmp = mxCalloc(ddimL*ddimK, sizeof(double));  
  y = mxCalloc(ddimL*ddimK, sizeof(double));
  
  plhs[0] = mxCreateDoubleMatrix(1,m, mxREAL);
  res = mxGetPr(plhs[0]);

  /* y = (hl'*knd) * kdd   (ddimL x ddimK) */
  dgemm(cht, chn, &ddimL, &ddimK, &n, &oneD, hl, &n, knd, &n, &nullD, tmp, &ddimL);
  dgemm(chn, chn, &ddimL, &ddimK, &ddimK, &oneD, tmp, &ddimL, kdd, &ddimK, &nullD, y, &ddimL);

  /* tmp = Ldd * y  (ddimL * ddimK) */
  dgemm(chn, chn, &ddimL, &ddimK, &ddimL, &oneD, ldd, &ddimL, y, &ddimL, &nullD, tmp, &ddimL);

  /* term1: vec( hl * tmp)' * dKnd */
  tmp1 = mxCalloc(n*ddimK, sizeof(double));
  dgemm(chn, chn, &n, &ddimK, &ddimL, &oneD, hl, &n, tmp, &ddimL, &nullD, tmp1, &n);
  dgemm(cht, chn, &oneI, &m, &nd, &oneD, tmp1, &nd, dknd, &nd, &nullD, res, &oneI);

  /* term2: vec( y' * tmp)' * dKdd 
     result is 2*term1 - term2 */
  nd = ddimK * ddimK;
  mxFree(tmp1);
  tmp1 = mxCalloc(nd,sizeof(double));
  dgemm(cht, chn, &ddimK, &ddimK, &ddimL, &oneD, y, &ddimL, tmp, &ddimL, &nullD, tmp1, &ddimK);
  mxFree(tmp);
  mxFree(y); 

  dgemm(cht, chn, &oneI, &m, &nd, &mOneD, tmp1, &nd, dkdd, &nd, &twoD, res, &oneI);

  mxFree(tmp1);

  return;
}
