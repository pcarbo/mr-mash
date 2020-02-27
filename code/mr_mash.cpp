// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <cmath>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// FUNCTION DECLARATIONS
// ---------------------
double ldmvnorm (const arma::vec& x, const arma::mat& S);

double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
			const mat& S0, vec& bhat, mat& S, vec& mu1,
			mat& S1);

// FUNCTION DEFINITIONS
// --------------------
// This is mainly used to test the bayes_mvr_ridge C++ function
// defined below.
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

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
			const mat& S0, vec& bhat, mat& S, vec& mu1,
			mat& S1) {
  unsigned int r = Y.n_cols;
  
  // Compute the least-squares estimate of the coefficients (bhat) and
  // the covariance of the standard error (S).
  double xx = norm(x);
  xx  *= xx;
  bhat = trans(Y)*x/xx;
  S    = V/xx;

  // Compute the posterior mean (mu1) and covariance (S1) assumig63ng a
  // multivariate normal prior with zero mean and covariance S0.
  mat I(r,r,fill::eye);

  // S1  <- S0 %*% solve(I + solve(S) %*% S0)
  // mu1 <- drop(S1 %*% solve(S,bhat))

  // Compute the log-Bayes factor.
  return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
}

// Compute the log-probability density of the multivariate normal
// distribution with zero mean and covariance matrix S.
double ldmvnorm (const arma::vec& x, const arma::mat& S) {
  double n = (double) x.n_elem;
  mat    L = chol(S,"lower");
  double d = norm(solve(L,x),2);
  return -(n*log(2*M_PI) + d*d)/2 - sum(log(L.diag()));
}
