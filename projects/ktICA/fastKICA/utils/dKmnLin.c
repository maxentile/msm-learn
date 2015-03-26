#include "mex.h"

/*
  dK = dKmn(Knm, inds, w, T, sigma)
  Knm: n x d kernel submatrix, the result is the derivative of this matrix
  inds: indices from incomplete Cholesky decomposition
  K is constructed on wT (row vector w of demixing matrix), nonsparse
  T is original data
  sigma: kernel width

  computes dvec(Knm)/dw

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

  double *k, *dK, *t, *w, *tdiff;
  double *inds;
  double sigma, dprod, oneD=-1.0;
  int n, m, nw, i, j, colind, oneI = 1, nm;

  k = mxGetPr(prhs[0]);
  inds = (double*) mxGetPr(prhs[1]);
  w = mxGetPr(prhs[2]);
  t = mxGetPr(prhs[3]);
  sigma = *mxGetPr(prhs[4]);
  sigma *= -sigma;
  sigma = 1/sigma;
  n = mxGetM(prhs[0]);  /* number of samples */
  m = mxGetN(prhs[0]);  /* number of indices (called d elsewhere) */
  nw = mxGetN(prhs[2]); /* length of w: number of sources */
  nm = n*m;

  plhs[0] = mxCreateDoubleMatrix(nm, nw, mxREAL);
  dK = mxGetPr(plhs[0]);

  tdiff = mxCalloc(nw, sizeof(double));

  for (i=0; i<m; ++i){ /* column index */
    colind = (int)(inds[i]) - 1;
    for (j=0; j<n; ++j){

      /* tdiff = tj - ti */
      dcopy_(&nw, &t[j*nw], &oneI, tdiff, &oneI);
      daxpy_(&nw, &oneD, &t[colind*nw], &oneI, tdiff, &oneI);

      /* dprod = w * (ti - tj) */
      dprod = ddot_(&nw, w, &oneI, tdiff, &oneI);

      /* dK( (i,j),:) = -K(i,j)/sigma * w * (tj - ti)(tj - ti)' */
      dcopy_(&nw, tdiff, &oneI, &dK[i*n + j], &nm);
      dprod *= k[i*n + j] * sigma;
      dscal_(&nw, &dprod, &dK[i*n + j], &nm);

    }
  }

  mxFree(tdiff);
  return;

}
