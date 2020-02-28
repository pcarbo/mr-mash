// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// FUNCTION DECLARATIONS
// ---------------------
double ldmvnorm (const arma::vec& x, const arma::mat& S);

double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
			const mat& S0, vec& bhat, mat& S, vec& mu1,
			mat& S1);

double bayes_mvr_mix (const vec& x, const mat& Y, const mat& V,
		      const vec& w0, const cube& S0, vec& mu1, mat& S1,
		      vec& w1);
  
// FUNCTION DEFINITIONS
// --------------------
// This is mainly used to test the bayes_mvr_ridge C++ function
// defined below. It is called in the same way as
// bayes_mvr_ridge_simple, e.g.,
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

// This is mainly used to test the bayes_mvr_mix C++ function
// defined below.
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

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
			const mat& S0, vec& bhat, mat& S, vec& mu1,
			mat& S1) {
  unsigned int r = Y.n_cols;
  
  // Compute the least-squares estimate of the coefficients (bhat) and
  // the covariance of the standard error (S).
  double xx = dot(x,x);
  bhat = trans(Y) * x/xx;
  S    = V/xx;

  // Compute the posterior mean (mu1) and covariance (S1) assumig63ng a
  // multivariate normal prior with zero mean and covariance S0.
  mat I(r,r,fill::eye);
  S1  = S0 * inv(I + solve(S,S0));
  mu1 = S1 * solve(S,bhat);

  // Compute the log-Bayes factor.
  return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
}

// Compare this to the R function bayes_mvr_mix_simple.
double bayes_mvr_mix (const vec& x, const mat& Y, const mat& V,
		      const vec& w0, const cube& S0, vec& mu1, mat& S1,
		      vec& w1) {
  unsigned int k = w0.n_elem;
  unsigned int r = Y.n_cols;
  vec  bhat(r);
  mat  S(r,r);
  vec  logbfmix(k);
  mat  mu1mix(r,k);
  cube S1mix(r,r,k);
  
  // Compute the quantities separately for each mixture component.
  for (unsigned int i = 0; i < k; i++) {
    logbfmix(i)    = bayes_mvr_ridge(x,Y,V,S0.slice(i),bhat,S,mu1,S1);
    mu1mix.col(i)  = mu1;
    S1mix.slice(i) = S1;
  }

  // Compute the posterior assignment probabilities for the latent
  // indicator variable.
  logbfmix += log(w0);
  // TO DO.
  
  // Compute the log-Bayes factor as a linear combination of the
  // individual Bayes factors for each mixture component.
  double u = max(logbfmix);
  return u + log(sum(exp(logbfmix - u)));
}

// Compute the log-probability density of the multivariate normal
// distribution with zero mean and covariance matrix S, omitting terms
// that do not depend on x or S.
double ldmvnorm (const arma::vec& x, const arma::mat& S) {
  mat    L = chol(S,"lower");
  double d = norm(solve(L,x),2);
  return -(d*d)/2 - sum(log(L.diag()));
}
