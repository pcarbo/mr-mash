// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// FUNCTION DECLARATIONS
// ---------------------
void softmax (vec& x);

double ldmvnorm (const vec& x, const mat& S);

double chol2ldet (const mat& R);

void inner_loop (const mat& X, mat& R, const mat& V,
                 const vec& w0, const cube& S0, mat& B);

void mr_mash_update (const mat& X, const mat& Y, const mat& V,
                     const vec& w0, const cube& S0, mat& B);

double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
                        const mat& S0, vec& bhat, mat& S, vec& mu1,
                        mat& S1);

double bayes_mvr_ridge_scaled_X (const vec& b, const mat& S0, const mat& S,
				 const mat& S1, const mat& SplusS0_chol,
				 const mat& S_chol, double ldetSplusS0_chol,
				 double ldet_chol, vec& mu1);

double bayes_mvr_ridge_centered_X (const mat& V, const vec& b, const mat& S, 
                                   const mat& S0, double xtx, const mat& V_chol,
                                   const mat& S_chol, const mat& U0, const vec& d,
                                   const mat& Q, vec& mu1, mat& S1);

double bayes_mvr_mix (const vec& x, const mat& Y, const mat& V,
                      const vec& w0, const cube& S0, vec& mu1, mat& S1,
                      vec& w1);

double bayes_mvr_mix_scaled_X (const vec& x, const mat& Y, const vec& w0,
                               const cube& S0, const mat& S, const cube& S1,
                               const cube& SplusS0_chol, const mat& S_chol,
                               const vec& ldetSplusS0_chol, double ldetS_chol,
                               vec& mu1_mix, mat& S1_mix, vec& w1);

double bayes_mvr_mix_centered_X (const vec& x, const mat& Y, const mat& V,
                                 const vec& w0, const cube& S0, double xtx, 
                                 const mat& V_chol, const cube& U0, const mat& d, 
                                 const cube& Q, vec& mu1_mix, mat& S1_mix, vec& w1);


// FUNCTION DEFINITIONS
// --------------------
// Perform a single pass of the co-ordinate ascent updates for the
// mr-mash model. These two calls in R should produce the same result:
// 
//   B <- mr_mash_update_simple(X,Y,B,V,w0,S0)
//   B <- mr_mash_update_rcpp(X,Y,B,V,w0,simplify2array(S0))
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat mr_mash_update_rcpp (const arma::mat& X, const arma::mat& Y,
                               const arma::mat& B0, const arma::mat& V,
                               const arma::vec& w0, const arma::cube& S0) {
  mat B = B0;
  mr_mash_update(X,Y,V,w0,S0,B);
  return B;
}

// This is mainly used to test the bayes_mvr_ridge C++ function
// defined below. It is called in the same way as bayes_mvr_ridge_simple,
// e.g.,
//
//    out1 <- bayes_mvr_ridge_simple(x,Y,V,S0)
//    out2 <- bayes_mvr_ridge_rcpp(x,Y,V,S0)
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_ridge_rcpp (const arma::vec& x, const arma::mat& Y,
                           const arma::mat& V, const arma::mat& S0) {
  unsigned int n = Y.n_cols;
  vec    bhat(n);
  vec    mu1(n);
  mat    S(n,n);
  mat    S1(n,n);
  double logbf = bayes_mvr_ridge(x,Y,V,S0,bhat,S,mu1,S1);
  return List::create(Named("bhat")  = bhat,
                      Named("S")     = S,
                      Named("mu1")   = mu1,
                      Named("S1")    = S1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_ridge_scaled_X C++ function
// defined below. It is called in the same way as bayes_mvr_ridge_simple,
// e.g.,
//
//    out1 <- bayes_mvr_ridge_simple(x,Y,V,S0)
//    out2 <- bayes_mvr_ridge_scaled_X_rcpp(b, S0, S, S1, SplusS0_chol, S_chol, ldetSplusS0_chol, ldetS_chol)
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_ridge_scaled_X_rcpp (const arma::vec& b, const arma::mat& S0,
				    const arma::mat& S, const arma::mat& S1, 
                                    const arma::mat& SplusS0_chol,
				    const arma::mat& S_chol,
				    double ldetSplusS0_chol,
				    double ldet_chol) {
  unsigned int r = b.n_elem;
  vec    mu1(r);
  double logbf = bayes_mvr_ridge_scaled_X(b, S0, S, S1, SplusS0_chol, S_chol, ldetSplusS0_chol, ldet_chol, mu1);
  return List::create(Named("mu1")   = mu1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_ridge_centered_X C++ function
// defined below. It is called in the same way as bayes_mvr_ridge_simple,
// e.g.,
//
//    out1 <- bayes_mvr_ridge_simple(x,Y,V,S0)
//    out2 <- bayes_mvr_ridge_centered_X_rcpp(V, b, S, S0, xtx, V_chol, S_chol, U0, d, Q)
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_ridge_centered_X_rcpp (const mat& V, const vec& b, const mat& S, 
                                      const mat& S0, double xtx, const mat& V_chol,
                                      const mat& S_chol, const mat& U0, const vec& d,
                                      const mat& Q) {
  unsigned int r = b.n_elem;
  vec    mu1(r);
  mat    S1(r,r);
  double logbf = bayes_mvr_ridge_centered_X(V, b, S, S0, xtx, V_chol, S_chol, U0, d, Q, mu1, S1);
  return List::create(Named("mu1")   = mu1,
                      Named("S1")   = S1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_mix C++ function defined
// below. It is called in the same way as bayes_mvr_mix_simple, except
// that input S0 is not a list of matrices, but rather an r x r x k
// "cube" (3-d array), storing the S0 matrices. This cube can easily
// be obtained from the list using the R function simplify2array,
// e.g.,
// 
//   out1 <- bayes_mvr_mix_simple(x,R,V,w0,S0)
//   out2 <- bayes_mvr_mix_rcpp(x,R,V,w0,simplify2array(S0))
//	       
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_mix_rcpp (const arma::vec& x, const arma::mat& Y,
                         const arma::mat& V, const arma::vec& w0,
                         const arma::cube& S0) {
  unsigned int r = Y.n_cols;
  unsigned int k = w0.n_elem;
  vec mu1(r);
  mat S1(r,r);
  vec w1(k);
  double logbf = bayes_mvr_mix(x,Y,V,w0,S0,mu1,S1,w1);
  return List::create(Named("mu1")   = mu1,
                      Named("S1")    = S1,
                      Named("w1")    = w1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_mix_scaled_X C++ function defined
// below. It is called in the same way as bayes_mvr_mix_simple, except
// that input S0 is not a list of matrices, but rather an r x r x k
// "cube" (3-d array), storing the S0 matrices. This cube can easily
// be obtained from the list using the R function simplify2array,
// e.g.,
// 
//   out1 <- bayes_mvr_mix_simple(x,R,V,w0,S0)
//   out2 <- bayes_mvr_mix_scaled_X_rcpp(X[,1], Y, w0, simplify2array(S0), S, 
//                                        simplify2array(S1), simplify2array(SplusS0_chol), 
//                                        S_chol, ldetSplusS0_chol, ldetS_chol)
//	       
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_mix_scaled_X_rcpp (const arma::vec& x, const arma::mat& Y,
                                  const arma::vec& w0, const arma::cube& S0,
                                  const arma::mat& S, const arma::cube& S1,
                                  const arma::cube& SplusS0_chol,
                                  const arma::mat& S_chol,
                                  const arma::vec& ldetSplusS0_chol,
                                  double ldetS_chol) {
  unsigned int r = Y.n_cols;
  unsigned int k = w0.n_elem;
  vec mu1_mix(r);
  mat S1_mix(r,r);
  vec w1(k);
  double logbf_mix = bayes_mvr_mix_scaled_X(x,Y,w0,S0,S,S1,SplusS0_chol,S_chol, 
                                            ldetSplusS0_chol, ldetS_chol, mu1_mix, S1_mix, w1);
  
  return List::create(Named("mu1")   = mu1_mix,
                      Named("S1")    = S1_mix,
                      Named("w1")    = w1,
                      Named("logbf") = logbf_mix);
}

// This is mainly used to test the bayes_mvr_mix_centered_X C++ function defined
// below. It is called in the same way as bayes_mvr_mix_simple, except
// that input S0 is not a list of matrices, but rather an r x r x k
// "cube" (3-d array), storing the S0 matrices. This cube can easily
// be obtained from the list using the R function simplify2array,
// e.g.,
// 
//   out1 <- bayes_mvr_mix_simple(x,R,V,w0,S0)
//   out2 <- bayes_mvr_mix_centered_X_rcpp(X[,1], Y, V, w0, simplify2array(S0), xtx, 
//                                        V_chol, simplify2array(U0), simplify2array(d),
//                                        simplify2array(Q))
//	       
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_mix_centered_X_rcpp (const arma::vec& x, const arma::mat& Y, const arma::mat& V,
                                    const arma::vec& w0, const arma::cube& S0, double xtx, 
                                    const arma::mat& V_chol, const arma::cube& U0, const arma::mat& d, 
                                    const arma::cube& Q) {
  unsigned int r = Y.n_cols;
  unsigned int k = w0.n_elem;
  vec mu1_mix(r);
  mat S1_mix(r,r);
  vec w1(k);
  double logbf_mix = bayes_mvr_mix_centered_X(x, Y, V, w0, S0, xtx, V_chol, U0, d, Q, mu1_mix, S1_mix, w1);
  
  return List::create(Named("mu1")   = mu1_mix,
                      Named("S1")    = S1_mix,
                      Named("w1")    = w1,
                      Named("logbf") = logbf_mix);
}



//Perform the inner loop
void inner_loop (const mat& X, mat& R, const mat& V,
                 const vec& w0, const cube& S0, mat& B) {
  unsigned int n = X.n_rows;
  unsigned int p = X.n_cols;
  unsigned int r = R.n_cols;
  unsigned int k = w0.n_elem;
  mat S1(r,r);
  vec x(n);
  vec b(r);
  vec w1(k);
  
  // Repeat for each predictor.
  for (unsigned int i = 0; i < p; i++) {
    x = X.col(i);
    b = trans(B.row(i));
    
    // Disregard the ith predictor in the expected residuals.
    R += x * trans(b);
    
    // Update the posterior of the regression coefficients for the ith
    // predictor.
    bayes_mvr_mix(x,R,V,w0,S0,b,S1,w1);
    B.row(i) = trans(b);
    
    // Update the expected residuals.
    R -= x * trans(b);
  }
}


// Compare this to the R function mr_mash_update_simple. Use
// mr_mash_update_rcpp to call this function from R.
void mr_mash_update (const mat& X, const mat& Y, const mat& V,
                     const vec& w0, const cube& S0, mat& B) {
  
  // Compute the expected residuals.
  mat R = Y - X * B;
  
  // Repeat for each predictor.
  inner_loop(X, R, V, w0, S0, B);
}

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
                        const mat& S0, vec& bhat, mat& S, vec& mu1,
                        mat& S1) {
  // unsigned int r = Y.n_cols;
  
  // Compute the least-squares estimate of the coefficients (bhat) and
  // the covariance of the standard error (S).
  double xx = dot(x,x);
  bhat = trans(Y) * x/xx;
  S    = V/xx;
  
  // Compute the posterior mean (mu1) and covariance (S1) assumig63ng a
  // multivariate normal prior with zero mean and covariance S0.
  // mat I(r,r,fill::eye);
  // S1  = S0 * inv(I + solve(S,S0));
  mat SplusS0_chol = chol(S+S0, "upper");
  S1 = S0*solve(trimatu(SplusS0_chol), solve(trimatl(trans(SplusS0_chol)), S));
  
  //mu1 = S1 * solve(S,bhat);
  mat S_chol = chol(S, "upper");
  mu1 = S1 * solve(trimatu(S_chol), solve(trimatl(trans(S_chol)), bhat));
  
  // Compute the log-Bayes factor.
  return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
}

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge_scaled_X (const vec& b, const mat& S0, const mat& S,
				 const mat& S1, const mat& SplusS0_chol,
				 const mat& S_chol, double ldetSplusS0_chol,
				 double ldetS_chol,
                                 vec& mu1) {
  // Compute the posterior mean (mu1) aassuming a multivariate 
  // normal prior with zero mean and covariance S0.
  mu1 = S1 * solve(trimatu(S_chol), solve(trimatl(trans(S_chol)), b));
  
  // Compute the log-Bayes factor.
  // return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
  return (ldetS_chol - ldetSplusS0_chol +
          dot(b, solve(trimatu(S_chol), solve(trimatl(trans(S_chol)), b))) - 
          dot(b, solve(trimatu(SplusS0_chol), solve(trimatl(trans(SplusS0_chol)), b))))/2;
}

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge_centered_X (const mat& V, const vec& b, const mat& S, 
                                   const mat& S0, double xtx, const mat& V_chol,
                                   const mat& S_chol, const mat& U0, const vec& d,
                                   const mat& Q, vec& mu1, mat& S1) {
  // Compute the posterior mean (mu1) and covariance (S1) assuming a multivariate 
  // normal prior with zero mean and covariance S0.
  mat D = diagmat(1/(1 + xtx * d));
  mat U1 = U0 * Q * D * trans(Q);
  S1 = trans(V_chol) * U1 * V_chol;
  
  mu1 = S1 * solve(trimatu(S_chol), solve(trimatl(trans(S_chol)), b));
  
  // Compute the log-Bayes factor.
  // return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
  mat SplusS0_chol = chol(S+S0, "upper");
  return (chol2ldet(S_chol) - chol2ldet(SplusS0_chol) +
          dot(b, solve(trimatu(S_chol), solve(trimatl(trans(S_chol)), b))) - 
          dot(b, solve(trimatu(SplusS0_chol), solve(trimatl(trans(SplusS0_chol)), b))))/2;
}

// Compare this to the R function bayes_mvr_mix_simple.
double bayes_mvr_mix (const vec& x, const mat& Y, const mat& V,
                      const vec& w0, const cube& S0, vec& mu1, mat& S1,
                      vec& w1) {
  unsigned int k = w0.n_elem;
  unsigned int r = Y.n_cols;
  vec  b(r);
  mat  S(r,r);
  vec  logbfmix(k);
  mat  mu1mix(r,k);
  cube S1mix(r,r,k);
  
  // Compute the quantities separately for each mixture component.
  for (unsigned int i = 0; i < k; i++) {
    logbfmix(i)    = bayes_mvr_ridge(x,Y,V,S0.slice(i),b,S,mu1,S1);
    mu1mix.col(i)  = mu1;
    S1mix.slice(i) = S1;
  }
  
  // Compute the posterior assignment probabilities for the latent
  // indicator variable.
  logbfmix += log(w0);
  w1 = logbfmix;
  softmax(w1);
  
  // Compute the posterior mean (mu1) and covariance (S1) of the
  // regression coefficients.
  S1.fill(0);
  mu1.fill(0);
  for (unsigned int i = 0; i < k; i++) {
    b    = mu1mix.col(i);
    mu1 += w1(i) * b;
    S1  += w1(i) * (S1mix.slice(i) + b * trans(b));
  }
  S1 -= mu1 * trans(mu1);
  
  // Compute the log-Bayes factor as a linear combination of the
  // individual Bayes factors for each mixture component.
  double u = max(logbfmix);
  return u + log(sum(exp(logbfmix - u)));
}

// Compare this to the R function bayes_mvr_mix_simple.
double bayes_mvr_mix_scaled_X (const vec& x, const mat& Y, const vec& w0,
                               const cube& S0, const mat& S, const cube& S1,
                               const cube& SplusS0_chol, const mat& S_chol,
                               const vec& ldetSplusS0_chol, double ldetS_chol,
                               vec& mu1_mix, mat& S1_mix, vec& w1) {
  unsigned int k = w0.n_elem;
  unsigned int r = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  mat mu1mix(r,k);
  vec logbfmix(k);
  vec mu1(r);
  
  // Compute the least-squares estimate.
  vec b = trans(Y)*x/(n-1);
  
  // Compute the quantities separately for each mixture component.
  for (unsigned int i = 0; i < k; i++) {
    logbfmix(i)    =  bayes_mvr_ridge_scaled_X(b, S0.slice(i), S, S1.slice(i), SplusS0_chol.slice(i), S_chol, 
             ldetSplusS0_chol(i), ldetS_chol, mu1);
    
    mu1mix.col(i)  = mu1;
  }
  
  // Compute the posterior assignment probabilities for the latent
  // indicator variable.
  logbfmix += log(w0);
  w1 = logbfmix;
  softmax(w1);
  
  // Compute the posterior mean (mu1) and covariance (S1_mix) of the
  // regression coefficients.
  S1_mix.fill(0);
  mu1_mix.fill(0);
  for (unsigned int i = 0; i < k; i++) {
    b    = mu1mix.col(i);
    mu1_mix += w1(i) * b;
    S1_mix  += w1(i) * (S1.slice(i) + b * trans(b));
  }
  S1_mix -= mu1_mix * trans(mu1_mix);
  
  // Compute the log-Bayes factor as a linear combination of the
  // individual Bayes factors for each mixture component.
  double u = max(logbfmix);
  return u + log(sum(exp(logbfmix - u)));
}

// Compare this to the R function bayes_mvr_mix_simple.
double bayes_mvr_mix_centered_X (const vec& x, const mat& Y, const mat& V,
                                 const vec& w0, const cube& S0, double xtx, 
                                 const mat& V_chol, const cube& U0, const mat& d, 
                                 const cube& Q, vec& mu1_mix, mat& S1_mix, vec& w1) {
  unsigned int k = w0.n_elem;
  unsigned int r = Y.n_cols;
  
  mat mu1mix(r,k);
  cube S1mix(r,r,k);
  vec logbfmix(k);
  vec mu1(r);
  mat S1(r,r);
  
  // Compute the least-squares estimate.
  vec b = trans(Y)*x/xtx;
  mat S = V/xtx;
  
  // Compute quantities needed for bayes_mvr_ridge_centered_X()
  mat S_chol = V_chol/sqrt(xtx);
  
  // Compute the quantities separately for each mixture component.
  for (unsigned int i = 0; i < k; i++) {
    logbfmix(i)    =  bayes_mvr_ridge_centered_X(V, b, S, S0.slice(i), xtx, V_chol, S_chol, 
                                                  U0.slice(i), d.col(i), Q.slice(i), mu1, S1);
    
    mu1mix.col(i)  = mu1;
    S1mix.slice(i) = S1;
  }
  
  // Compute the posterior assignment probabilities for the latent
  // indicator variable.
  logbfmix += log(w0);
  w1 = logbfmix;
  softmax(w1);
  
  // Compute the posterior mean (mu1) and covariance (S1_mix) of the
  // regression coefficients.
  S1_mix.fill(0);
  mu1_mix.fill(0);
  for (unsigned int i = 0; i < k; i++) {
    b    = mu1mix.col(i);
    mu1_mix += w1(i) * b;
    S1_mix  += w1(i) * (S1mix.slice(i) + b * trans(b));
  }
  S1_mix -= mu1_mix * trans(mu1_mix);
  
  // Compute the log-Bayes factor as a linear combination of the
  // individual Bayes factors for each mixture component.
  double u = max(logbfmix);
  return u + log(sum(exp(logbfmix - u)));
}

// Compute the softmax of x, and return the result in x. Guard against
// underflow or overflow by adjusting the entries of x so that the
// largest value is zero.
void softmax (vec& x) {
  x -= max(x);
  x  = exp(x);
  x /= sum(x);
}

// Compute the log-probability density of the multivariate normal
// distribution with zero mean and covariance matrix S, omitting terms
// that do not depend on x or S.
double ldmvnorm (const arma::vec& x, const arma::mat& S) {
  mat    L = chol(S,"lower");
  double d = norm(solve(L,x),2);
  return -(d*d)/2 - sum(log(L.diag()));
}

// Compute the log determinant from Cholesky decomposed matrix
double chol2ldet (const mat& R){
  return log(prod(R.diag()))*2;
}









// PETER TO DEAL WITH LIST

mat inner_loop_general (const mat& X, const mat& rbar, const mat& mu1, const mat& V,
                         const vec& w0, const cube& S0, const List& precomp_quants, 
                         bool standardize, mat& mu1_new, cube& S1, mat& w1);


//Perform the inner loop
mat inner_loop_general (const mat& X, const mat& rbar, const mat& mu1, const mat& V,
                         const vec& w0, const cube& S0, const List& precomp_quants, 
                         bool standardize, mat& mu1_new, cube& S1, mat& w1) {
  unsigned int n = X.n_rows;
  unsigned int p = X.n_cols;
  unsigned int r = rbar.n_cols;
  unsigned int k = w0.n_elem;
  vec x(n);
  mat rbar_j(n,r);
  vec mu1_j(r);
  vec mu1_mix(r);
  mat S1_mix(r,r);
  vec w1_mix(k);
  mat rbar_new = rbar;
  
  // Repeat for each predictor.
  for (unsigned int j = 0; j < p; j++) {
    x = X.col(j);
    mu1_j = trans(mu1.row(j));
    
    // Disregard the ith predictor in the expected residuals.
    rbar_j = rbar_new + (x * trans(mu1_j));
    
    // Update the posterior quantities for the jth
    // predictor.
    if(standardize){
      mat S = as<mat>(precomp_quants["S"]);
      cube S1 = as<cube>(precomp_quants["S1"]);
      cube SplusS0_chol = as<cube>(precomp_quants["SplusS0_chol"]);
      mat S_chol = as<mat>(precomp_quants["S_chol"]);
      vec ldetSplusS0_chol = as<vec>(precomp_quants["ldetSplusS0_chol"]);
      double ldetS_chol = as<double>(precomp_quants["ldetS_chol"]);
      
      bayes_mvr_mix_scaled_X(x, rbar_j, w0, S0, S, S1, SplusS0_chol, S_chol, 
                             ldetSplusS0_chol, ldetS_chol, mu1_mix, S1_mix, w1_mix);
    } else {
      vec xtx = as<vec>(precomp_quants["xtx"]);
      double xtx_j = xtx(j);
      mat V_chol = as<mat>(precomp_quants["V_chol"]);
      cube U0 = as<cube>(precomp_quants["U0"]);
      mat d = as<mat>(precomp_quants["d"]);
      cube Q = as<cube>(precomp_quants["Q"]);
      
      bayes_mvr_mix_centered_X(x, rbar_j, V, w0, S0, xtx_j, V_chol, U0, d, Q,
                               mu1_mix, S1_mix, w1_mix);
    }
    
    mu1_new.row(j) = trans(mu1_mix);
    S1.slice(j) = S1_mix;
    w1.row(j) = trans(w1_mix);
    
    // Update the expected residuals.
    rbar_new = rbar_j - (x * trans(mu1_mix));
  }
  
  return rbar_new;
}


// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List inner_loop_general_rcpp (const arma::mat& X, const arma::mat& rbar, const mat& mu1, 
                              const arma::mat& V, const arma::vec& w0, const arma::cube& S0, 
                              const List& precomp_quants, bool standardize) {
  unsigned int r = rbar.n_cols;
  unsigned int n = rbar.n_rows;
  unsigned int p = X.n_cols;
  unsigned int k = w0.n_elem;
  mat mu1_new(p,r);
  cube S1(r,r,p);
  mat w1(p,k);
  mat rbar_new = inner_loop_general(X, rbar, mu1, V, w0, S0, precomp_quants, standardize, mu1_new, S1, w1);
  
  return List::create(Named("rbar")  = rbar_new,
                      Named("mu1")   = mu1_new,
                      Named("S1")    = S1,
                      Named("w1")    = w1);
}
