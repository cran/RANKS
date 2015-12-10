#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 
  Function to compute the minimum in a vector
  INPUT:
  v: vector of double
  k: length of the vector v
  OUTPUT
  minimum of v
*/
double min(double *v, int k) {
 register int i;
 register double z = v[0];
 for (i=1; i < k; i++) {
	  if (v[i]<z) 
	    z = v[i];	
 }
 return(z);
}

/* 
  Function to find the position (index) of  the minimum in a vector
  INPUT:
  v: vector of double
  k: length of the vector v
  OUTPUT
  position (index of v) of the minimum
*/
int pos_min(double *v, int k) {
 register int i;
 register double z = v[0];
 register int pos = 0;
 for (i=1; i < k; i++) {
	  if (v[i]<z) {
	    z = v[i];
		pos = i;
	  }	
 }
 return(pos);
}

/* 
  select_top: Function to select the first k largest elements in a vector
  INPUT:
  score_vect: vector containing the scores
  n: number scores (dimension of score_vect)
  k: number of the top scores to be selected
  OUTPUT
  selected_scores: vector of the top scores selected (length equal to k)
*/
void select_top(double *score_vect, double *selected_scores, int *n, int *k) {
	register int i, x;
	double min_selected;
	for (i=0; i < *k; i++) {
	  selected_scores[i] = score_vect[i];	
	}
	min_selected = min(selected_scores, *k);
	
	for(i = *k; i < *n; i++) {
	  if (score_vect[i] > min_selected) {
	    x = pos_min(selected_scores, *k);
		selected_scores[x] = score_vect[i];
		min_selected = min(selected_scores, *k);
	  }
	}
}


/* 
  norm_lapl_graph: Normalized graph Laplacian.
  Given an adjacency matrix of a graph, it computes the corresponding Normalized graph Laplacian
  INPUT:
  W: pointer to a symmetric matrix
  diag: pointer to a vector representing the diagonal of the matrix D^-1/2, where
  D_ii = \sum_j W_ij and D_ij = 0 if i!=j
  n: dimension of the square matrix W
  OUTPUT
  W is the normalized matrix
*/
void  norm_lapl_graph(double * W, double * diag, int * n){
  register int i, j, k, m;
  double x;
  m = *n;
  for (i=0; i <m; i++) {
    x = diag[i];
    for (j=0; j <m; j++){
	   k = j*m + i;	   
	   W[k] = W[k] * x;	
	}
  }
  
  for (i=0; i <m; i++) {
    for (j=0; j <m; j++){ 
	   x = diag[j];
	   k = j*m + i;	   
	   W[k] = W[k] * x;
	}
  }
}


double distnorm2(double* x, double* y, int n) {
  register int i;
  register double v;
  double z = 0;
  for (i=0; i<n; i++) {
    v = x[i] - y[i];
    z += (v*v);
  }
  return(z);  
}

/* 
  gaussian_kernel: Function to compute the Gram matrix of a gaussian kernel.
  Given a feature matrix, it computes the corresponding gaussian kernel
  INPUT:
  K: pointer to the kernel matrix
  W: pointer to the feature matrix. Rows are examples and columns features
  sigma: sigma parameter of the kernel
  n : number of rows of W
  m : number of columns of W
  OUTPUT
  W is the normalized matrix
*/
void  gaussian_kernel(double * K, double * W, double * sigma, int * n, int * m){
  register int i, j, k;
  double x, z;
  double s = *sigma * *sigma;
  
  int n_ex = *n;
  int n_f = *m; 
  
  for (i=0; i<(n_ex-1); i++) 
    for(j=i+1; j<n_ex; j++) {
	  z = 0;
	  for(k=0; k<n_f; k++) {
	    x = W[i + n_ex * k] - W[j + n_ex * k];
		z += x*x;
	  }
	  K[j + i*n_ex] = K[i + j*n_ex] = exp(-z/s);	  
	}
  for (i=0; i<n_ex; i++)  
	K[i + i*n_ex] = 1; 
}


/* 
  laplacian_kernel: Function to compute the Gram matrix of a laplacian kernel.
  Given a feature matrix, it computes the corresponding laplacian kernel
  INPUT:
  K: pointer to the kernel matrix
  W: pointer to the feature matrix. Rows are examples and columns features
  sigma: sigma parameter of the kernel
  n : number of rows of W
  m : number of columns of W
  OUTPUT
  W is the normalized matrix
*/
void  laplacian_kernel(double * K, double * W, double * sigma, int * n, int * m){
  register int i, j, k;
  double x, z;
  double s = *sigma;
  
  int n_ex = *n;
  int n_f = *m; 
  
  for (i=0; i<(n_ex-1); i++) 
    for(j=i+1; j<n_ex; j++) {
	  z = 0;
	  for(k=0; k<n_f; k++) {
	    x = W[i + n_ex * k] - W[j + n_ex * k];
		z += x*x;
	  }
	  K[j + i*n_ex] = K[i + j*n_ex] = exp(-sqrt(z)/s);	  
	}
  for (i=0; i<n_ex; i++)  
	K[i + i*n_ex] = 1; 
}


/* 
  cauchy_kernel: Function to compute the Gram matrix of a Cauchy kernel.
  Given a feature matrix, it computes the corresponding Cauchy kernel
  INPUT:
  K: pointer to the kernel matrix
  W: pointer to the feature matrix. Rows are examples and columns features
  sigma: sigma parameter of the kernel
  n : number of rows of W
  m : number of columns of W
  OUTPUT
  W is the normalized matrix
*/
void  cauchy_kernel(double * K, double * W, double * sigma, int * n, int * m){
  register int i, j, k;
  double x, z;
  double s = *sigma;
  
  int n_ex = *n;
  int n_f = *m; 
  
  for (i=0; i<(n_ex-1); i++) 
    for(j=i+1; j<n_ex; j++) {
	  z = 0;
	  for(k=0; k<n_f; k++) {
	    x = W[i + n_ex * k] - W[j + n_ex * k];
		z += x*x;
	  }
	  K[j + i*n_ex] = K[i + j*n_ex] = 1/(1 + z/s);	  
	}
  for (i=0; i<n_ex; i++)  
	K[i + i*n_ex] = 1; 
}

/* 
  inv_mq_kernel: Function to compute the Gram matrix of an inverse multiquadric kernel.
  Given a feature matrix, it computes the corresponding inverse multiquadric kernel
  INPUT:
  K: pointer to the kernel matrix
  W: pointer to the feature matrix. Rows are examples and columns features
  v: constant parameter of the kernel (v > 0)
  n : number of rows of W
  m : number of columns of W
  OUTPUT
  W is the normalized matrix
*/
void  inv_mq_kernel(double * K, double * W, double * v, int * n, int * m){
  register int i, j, k;
  double x, z;
  double cost = *v * *v;
  double value_diag = 1 / *v;
  
  int n_ex = *n;
  int n_f = *m; 
  
  for (i=0; i<(n_ex-1); i++) 
    for(j=i+1; j<n_ex; j++) {
	  z = 0;
	  for(k=0; k<n_f; k++) {
	    x = W[i + n_ex * k] - W[j + n_ex * k];
		z += x*x;
	  }
	  K[j + i*n_ex] = K[i + j*n_ex] = 1/(sqrt(z + cost)); 
	}
  for (i=0; i<n_ex; i++)  
	K[i + i*n_ex] = value_diag; 
}


/* 
  poly_kernel: Function to compute the Gram matrix of a polynomial kernel.
  Given a feature matrix, it computes the corresponding polynomial kernel
  INPUT:
  K: pointer to the kernel matrix
  W: pointer to the feature matrix. Rows are examples and columns features
  degree: degree of the polynomial
  scale : scaling factor
  v : constant value of the polynomial
  n : number of rows of W
  m : number of columns of W
  OUTPUT
  W is the normalized matrix
*/


void  poly_kernel(double * K, double * W, int * degree, double* scale, double * v, int * n, int * m){
  register int i, j, k;
  double z;
  
  register int n_ex = *n;
  int n_f = *m; 
  double sc = *scale;
  double offset = *v;
  int deg = *degree;
    
  for (i=0; i<n_ex; i++) 
    for(j=i; j<n_ex; j++) {
	  z = 0;
	  for(k=0; k<n_f; k++) 
	    z += W[i + n_ex * k] * W[j + n_ex * k];
	  K[j + i*n_ex] = K[i + j*n_ex] = pow(sc * z + offset, deg);	  
	}
}


int compare_doubles (const void *a, const void *b)   {
       const double *da = (const double *) a;
       const double *db = (const double *) b;    
       return (*da > *db) - (*da < *db);
}



/* 
  wsld: Function to compute the WSLD score
  INPUT:
  res: result of the WSLD score
  x: pointer to a vector with the values of the positive elements
  n: length of the vector x
  d : integer corresponding to the D parameter of the WSLD score
  OUTPUT
  res is the computed score
*/
void  wsld2(double * res, double * x, int * n, int * d){
  register int i, m, j;
  double y = 0;
  m = *n;
  qsort(x, m, sizeof(double), compare_doubles);
  y = x[m-1];
  for (i=m-2; i >=0; i--) {
    if (x[i]==0) break;
	j = m-i-1;
    y += x[i]/(*d * j);
  }
  *res = y;
}
 
/* 
  do_wsld_scores_from_matrix : Function to compute the WSLD scores from a matrix
  INPUT:
  scores: vector of doubles with  the results  of the WSLD score
  K: pointer to the subkernel matrix. It may be also part of a symmetric matrix. Columns correspond to examples for which we 
     compute the WSLD score, and rows correspond to the positive elements (in other words the ith column stores the positive elements 
	 for the ith example) 
  m: number of examples (columns)
  n : number of rows (positive examples)
  d : integer corresponding to the D parameter of the WSLD score
  OUTPUT
  scores the computed score
*/
void do_wsld_scores_from_matrix (double * scores, double * K, int *m, int *n, int *d) {

 register int i;
  
 int n_ex = *m; 
 int n_f = *n;
 
 for (i=0; i < n_ex ; i++) 
   wsld2(scores + i, K+i*n_f, n, d);
}
