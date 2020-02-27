// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// FUNCTION DECLARATIONS
// ---------------------
double bayes_mvr_ridge_simple (const vec& x, const mat& Y, const mat& V,
			       const mat& S0, vec& bhat, mat& S, vec& mu1,
			       mat& S1);

// FUNCTION DEFINITIONS
// --------------------
// This is mainly used to test the bayes_mvr_ridge_simple C++ function
// defined below.
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_ridge_simple_rcpp (const arma::vec& x, const arma::mat& Y,
				  const arma::mat& V, const arma::mat& S0) {
  unsigned int n = Y.n_cols;
  vec    bhat(n);
  vec    mu1(n);
  mat    S(n,n);
  mat    S1(n,n);
  double logbf = bayes_mvr_ridge_simple(x,Y,V,S0,bhat,S,mu1,S1);
  return List::create(Named("bhat")  = bhat,
                      Named("S")     = S,
                      Named("mu1")   = mu1,
                      Named("S1")    = S1,
                      Named("logbf") = logbf);
}

// Compare this to the R function with the same name.
double bayes_mvr_ridge_simple (const vec& x, const mat& Y, const mat& V,
			       const mat& S0, vec& bhat, mat& S, vec& mu1,
			       mat& S1) {
  unsigned int r = Y.n_cols;
  
  // Compute the least-squares estimate of the coefficients (bhat) and
  // the covariance of the standard error (S).
  double xx = norm(x);
  xx  *= xx;
  bhat = trans(Y)*x/xx;
  S    = V/xx;

  // Compute the posterior mean (mu1) and covariance (S1) assuming a
  // multivariate normal prior with zero mean and covariance S0.
  mat I(r,r,fill::eye);
  
  // S1  <- S0 %*% solve(I + solve(S) %*% S0)
  // mu1 <- drop(S1 %*% solve(S,bhat))

  // Compute the log-Bayes factor.
  // logbf <- ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S)
  return 0;
}
