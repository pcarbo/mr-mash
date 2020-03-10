// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// TYPE DEFINITIONS
// ----------------
// A list of precomputed quantities that are invariant to any updates
// to the mr-mash model parameters.
struct mr_mash_precomputed_quantities {
  const mat  S;
  const mat  S_chol;
  const cube S1;
  const cube SplusS0_chol;

  // This is used to create a mr_mash_precomputed_quantities object.
  mr_mash_precomputed_quantities (const mat& S, const mat& S_chol,
				  const cube& S1, const cube& SplusS0_chol) :
    S(S), S_chol(S_chol), S1(S1), SplusS0_chol(SplusS0_chol) { };
};

// INLINE FUNCTION DEFINITIONS
// ---------------------------
// Solve for x in U'x = b by forward substitution.
inline vec forwardsolve (const mat& U, const vec& b) {
  return solve(trimatl(trans(U)),b);  
}

// Solve for x in Ux = b by back substitution.
inline vec backsolve (const mat& U, const vec& b) {
  return solve(trimatu(U),b);
}

// FUNCTION DECLARATIONS
// ---------------------
void mr_mash_update (const mat& X, const mat& Y, const mat& V,
                     const vec& w0, const cube& S0, mat& B);

void inner_loop (const mat& X, mat& R, const mat& V, const vec& w0,
		 const cube& S0, mat& B);

void inner_loop_general (const mat& X, mat& Rbar, mat& mu1, const mat& V,
			 const vec& w0, const cube& S0,
			 const List& precomp_quants, bool standardize,
			 cube& S1, mat& w1);

double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
                        const mat& S0, vec& bhat, mat& S, vec& mu1,
                        mat& S1);

double bayes_mvr_ridge_scaled_X (const vec& b, const mat& S0, const mat& S,
				 const mat& S1, const mat& SplusS0_chol,
				 const mat& S_chol, double ldetSplusS0_chol,
				 double ldet_chol, vec& mu1);

double bayes_mvr_ridge_centered_X (const mat& V, const vec& b, const mat& S, 
                                   const mat& S0, double xtx,
				   const mat& V_chol, const mat& S_chol,
				   const mat& U0, const vec& d, const mat& Q,
				   vec& mu1, mat& S1);

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
                                 const mat& V_chol, const cube& U0,
				 const mat& d, const cube& Q, vec& mu1_mix,
				 mat& S1_mix, vec& w1);

void softmax (vec& x);

double ldmvnorm (const vec& x, const mat& S);

double ldmvnormdiff (const vec& x, const mat& S_chol,
		     const mat& SplusS0_chol);

double chol2ldet (const mat& R);

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

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List inner_loop_general_rcpp (const arma::mat& X, arma::mat& Rbar, mat& mu1,
			      const arma::mat& V, const arma::vec& w0,
			      const arma::cube& S0, const List& precomp_quants,
			      bool standardize) {
  unsigned int r = Rbar.n_cols;
  unsigned int p = X.n_cols;
  unsigned int k = w0.n_elem;
  cube S1(r,r,p);
  mat  w1(p,k);
  mat  mu1_new  = mu1;
  mat  Rbar_new = Rbar;
  inner_loop_general(X, Rbar_new, mu1_new, V, w0, S0, precomp_quants,
		     standardize, S1, w1);
  return List::create(Named("rbar") = Rbar_new,
                      Named("mu1")  = mu1_new,
                      Named("S1")   = S1,
                      Named("w1")   = w1);
}

// This is mainly used to test the bayes_mvr_ridge C++ function
// defined below. It is called in the same way as bayes_mvr_ridge_simple,
// e.g.,
//
//   out1 <- bayes_mvr_ridge_simple(x,Y,V,S0)
//   out2 <- bayes_mvr_ridge_rcpp(x,Y,V,S0)
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
//    out2 <- bayes_mvr_ridge_scaled_X_rcpp(b, S0, S, S1, SplusS0_chol, S_chol,
//                                          ldetSplusS0_chol, ldetS_chol)
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
  double logbf = bayes_mvr_ridge_scaled_X(b, S0, S, S1, SplusS0_chol, S_chol,
					  ldetSplusS0_chol, ldet_chol, mu1);
  return List::create(Named("mu1")   = mu1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_ridge_centered_X C++ function
// defined below. It is called in the same way as bayes_mvr_ridge_simple,
// e.g.,
//
//    out1 <- bayes_mvr_ridge_simple(x,Y,V,S0)
//    out2 <- bayes_mvr_ridge_centered_X_rcpp(V, b, S, S0, xtx, V_chol, 
//                                            S_chol, U0, d, Q)
// 
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_ridge_centered_X_rcpp (const mat& V, const vec& b, const mat& S,
				      const mat& S0, double xtx,
				      const mat& V_chol, const mat& S_chol,
				      const mat& U0, const vec& d,
				      const mat& Q) {
  unsigned int r = b.n_elem;
  vec    mu1(r);
  mat    S1(r,r);
  double logbf = bayes_mvr_ridge_centered_X(V, b, S, S0, xtx, V_chol,
					    S_chol, U0, d, Q, mu1, S1);
  return List::create(Named("mu1")   = mu1,
                      Named("S1")   = S1,
                      Named("logbf") = logbf);
}

// This is mainly used to test the bayes_mvr_mix C++ function defined
// below. It is called in the same way as bayes_mvr_mix_simple, except
// that input S0 is not a list of matrices, but rather an r x r x k
// "cube" (3-d array), storing the S0 matrices. This cube can easily
// be obtained from the list using the R function simplify2array, e.g.,
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
// be obtained from the list using the R function simplify2array, e.g.,
// 
//   out1 <- bayes_mvr_mix_simple(x,R,V,w0,S0)
//   out2 <- bayes_mvr_mix_scaled_X_rcpp(X[,1], Y, w0, simplify2array(S0), S, 
//             simplify2array(S1), simplify2array(SplusS0_chol), S_chol,
//             ldetSplusS0_chol, ldetS_chol)
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
  double logbf_mix = bayes_mvr_mix_scaled_X(x,Y,w0,S0,S,S1,SplusS0_chol,
					    S_chol, ldetSplusS0_chol,
					    ldetS_chol, mu1_mix, S1_mix, w1);
  return List::create(Named("mu1")   = mu1_mix,
                      Named("S1")    = S1_mix,
                      Named("w1")    = w1,
                      Named("logbf") = logbf_mix);
}

// This is mainly used to test the bayes_mvr_mix_centered_X C++
// function defined below. It is called in the same way as
// bayes_mvr_mix_simple, except that input S0 is not a list of
// matrices, but rather an r x r x k "cube" (3-d array), storing the
// S0 matrices. This cube can easily be obtained from the list using
// the R function simplify2array, e.g.,
// 
//   out1 <- bayes_mvr_mix_simple(x,R,V,w0,S0)
//   out2 <- bayes_mvr_mix_centered_X_rcpp(X[,1], Y, V, w0,
//             simplify2array(S0), xtx, V_chol, simplify2array(U0),
//             simplify2array(d), simplify2array(Q))
//	       
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List bayes_mvr_mix_centered_X_rcpp (const arma::vec& x, const arma::mat& Y,
				    const arma::mat& V, const arma::vec& w0,
				    const arma::cube& S0, double xtx, 
                                    const arma::mat& V_chol,
				    const arma::cube& U0, const arma::mat& d, 
                                    const arma::cube& Q) {
  unsigned int r = Y.n_cols;
  unsigned int k = w0.n_elem;
  vec mu1_mix(r);
  mat S1_mix(r,r);
  vec w1(k);
  double logbf_mix = bayes_mvr_mix_centered_X(x, Y, V, w0, S0, xtx, V_chol,
					      U0, d, Q, mu1_mix, S1_mix, w1);
  return List::create(Named("mu1")   = mu1_mix,
                      Named("S1")    = S1_mix,
                      Named("w1")    = w1,
                      Named("logbf") = logbf_mix);
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

//Perform the inner loop.
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

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge (const vec& x, const mat& Y, const mat& V,
                        const mat& S0, vec& bhat, mat& S, vec& mu1,
                        mat& S1) {
  
  // Compute the least-squares estimate of the coefficients (bhat) and
  // the covariance of the standard error (S).
  double xx = dot(x,x);
  bhat = trans(Y) * x/xx;
  S    = V/xx;
  
  // Compute the posterior mean (mu1) and covariance (S1) assumig63ng a
  // multivariate normal prior with zero mean and covariance S0.
  // This is equivalent C++ code (simpler, but slower):
  // 
  //   mat I(r,r,fill::eye)
  //   S1  = S0 * inv(I + solve(S,S0))
  //   mu1 = S1 * solve(S,bhat)
  //
  mat SplusS0_chol = chol(S + S0, "upper");
  mat S_chol       = chol(S, "upper");
  S1  = S0 * backsolve(SplusS0_chol, forwardsolve(SplusS0_chol, S));
  mu1 = S1 * backsolve(S_chol, forwardsolve(S_chol, bhat));
  
  // Compute the log-Bayes factor.
  return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S);
}

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge_scaled_X (const vec& b, const mat& S0, const mat& S,
				 const mat& S1, const mat& SplusS0_chol,
				 const mat& S_chol, double ldetSplusS0_chol,
				 double ldetS_chol, vec& mu1) {
  
  // Compute the posterior mean (mu1) aassuming a multivariate 
  // normal prior with zero mean and covariance S0.
  mu1 = S1 * backsolve(S_chol, forwardsolve(S_chol, b));
  
  // Compute the log-Bayes factor. This should give the same result as:
  //
  //   ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S)
  //
  return ldmvnormdiff(b,S_chol,SplusS0_chol);
}

// Compare this to the R function bayes_mvr_ridge_simple.
double bayes_mvr_ridge_centered_X (const mat& V, const vec& b, const mat& S, 
                                   const mat& S0, double xtx,
				   const mat& V_chol, const mat& S_chol,
				   const mat& U0, const vec& d, const mat& Q,
				   vec& mu1, mat& S1) {
  
  // Compute the posterior mean (mu1) and covariance (S1) assuming a
  // multivariate normal prior with zero mean and covariance S0.
  mat D  = diagmat(1/(1 + xtx * d));
  mat U1 = U0 * Q * D * trans(Q);
  S1  = trans(V_chol) * U1 * V_chol;
  mu1 = S1 * backsolve(S_chol, forwardsolve(S_chol, b));
  
  // Compute the log-Bayes factor.
  // return ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S)
  return ldmvnormdiff(b,S_chol,chol(S + S0, "upper"));
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
    logbfmix(i) = bayes_mvr_ridge_scaled_X(b, S0.slice(i), S, S1.slice(i),
					   SplusS0_chol.slice(i), S_chol, 
					   ldetSplusS0_chol(i), ldetS_chol,
					   mu1);
    mu1mix.col(i) = mu1;
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
                                 const mat& V_chol, const cube& U0,
				 const mat& d, const cube& Q, vec& mu1_mix,
				 mat& S1_mix, vec& w1) {
  unsigned int k = w0.n_elem;
  unsigned int r = Y.n_cols;
  
  mat  mu1mix(r,k);
  cube S1mix(r,r,k);
  vec  logbfmix(k);
  vec  mu1(r);
  mat  S1(r,r);
  
  // Compute the least-squares estimate.
  vec b = trans(Y)*x/xtx;
  mat S = V/xtx;
  
  // Compute quantities needed for bayes_mvr_ridge_centered_X()
  mat S_chol = V_chol/sqrt(xtx);
  
  // Compute the quantities separately for each mixture component.
  for (unsigned int i = 0; i < k; i++) {
    logbfmix(i) = bayes_mvr_ridge_centered_X(V, b, S, S0.slice(i), xtx, V_chol,
					     S_chol, U0.slice(i), d.col(i),
					     Q.slice(i), mu1, S1);
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
double ldmvnorm (const vec& x, const mat& S) {
  mat    L = chol(S,"lower");
  double d = norm(solve(L,x),2);
  return -(d*d)/2 - sum(log(L.diag()));
}

// Compute the difference of two multivariate normal log-densities,
//
//   ldmvnorm(x,S0 + S) - ldmvnorm(x,S)
//
// where S_chol is the right-hand Cholesky factor of S, and
// SplusS0_chol is the right-hand Cholesky factor of S + S0 (also an
// upper triangular matrix).
double ldmvnormdiff (const vec& x, const mat& S_chol,
		     const mat& SplusS0_chol) {
  return (chol2ldet(S_chol) -
	  chol2ldet(SplusS0_chol) +
          dot(x, backsolve(S_chol, forwardsolve(S_chol, x))) - 
          dot(x, backsolve(SplusS0_chol, forwardsolve(SplusS0_chol, x))))/2;
}

// Compute the log determinant from Cholesky decomposed matrix
double chol2ldet (const mat& R) {
  return 2*sum(log(R.diag()));
}

//Perform the inner loop
void inner_loop_general (const mat& X, mat& Rbar, mat& mu1, const mat& V,
			 const vec& w0, const cube& S0,
			 const List& precomp_quants, bool standardize,
			 cube& S1, mat& w1) {
  unsigned int n = X.n_rows;
  unsigned int p = X.n_cols;
  unsigned int r = Rbar.n_cols;
  unsigned int k = w0.n_elem;
  vec x(n);
  mat Rbar_j(n,r);
  vec mu1_j(r);
  vec mu1_mix(r);
  mat S1_mix(r,r);
  vec w1_mix(k);
  
  // Repeat for each predictor.
  for (unsigned int j = 0; j < p; j++) {
    x = X.col(j);
    mu1_j = trans(mu1.row(j));
    
    // Disregard the ith predictor in the expected residuals.
    Rbar_j = Rbar + (x * trans(mu1_j));
    
    // Update the posterior quantities for the jth
    // predictor.
    if(standardize){
      mat S                = as<mat>(precomp_quants["S"]);
      cube S1              = as<cube>(precomp_quants["S1"]);
      cube SplusS0_chol    = as<cube>(precomp_quants["SplusS0_chol"]);
      mat S_chol           = as<mat>(precomp_quants["S_chol"]);
      vec ldetSplusS0_chol = as<vec>(precomp_quants["ldetSplusS0_chol"]);
      double ldetS_chol    = as<double>(precomp_quants["ldetS_chol"]);
      
      bayes_mvr_mix_scaled_X(x, Rbar_j, w0, S0, S, S1, SplusS0_chol, S_chol, 
                             ldetSplusS0_chol, ldetS_chol, mu1_mix, S1_mix,
			     w1_mix);
    } else {
      vec xtx      = as<vec>(precomp_quants["xtx"]);
      double xtx_j = xtx(j);
      mat V_chol   = as<mat>(precomp_quants["V_chol"]);
      cube U0      = as<cube>(precomp_quants["U0"]);
      mat d        = as<mat>(precomp_quants["d"]);
      cube Q       = as<cube>(precomp_quants["Q"]);
      
      bayes_mvr_mix_centered_X(x, Rbar_j, V, w0, S0, xtx_j, V_chol, U0, d, Q,
                               mu1_mix, S1_mix, w1_mix);
    }
    
    mu1.row(j)  = trans(mu1_mix);
    S1.slice(j) = S1_mix;
    w1.row(j)   = trans(w1_mix);
    
    // Update the expected residuals.
    Rbar = Rbar_j - (x * trans(mu1_mix));
  }
}






// PRELIMINARY VERSION OF FUNCTIONS NEEDED TO COMPUTE THE ELBO AND UPDATE V.
// THEY COMPILE BUT I DON'T KNOW IF THEY RETURN THE CORRECT RESULTS.
// NEED TO BE TESTED ONCE THE INNER LOOP IS IMPLEMENTED IN THE PACKAGE
// AND THEN ORGANIZED PROPERLY WITHIN THIS SCRIPT.

// Function to compute some terms of the ELBO
void compute_ELBO_terms (double& var_part_tr_wERSS, double& neg_KL, double x_j,
                         const mat& rbar_j, double logbf, const mat& mu1, const mat& S1, 
                         double xtx, const mat& Vinv);

void compute_ELBO_terms (double& var_part_tr_wERSS, double& neg_KL, double x_j,
                         const mat& rbar_j, double logbf, const mat& mu1, const mat& S1, 
                         double xtx, const mat& Vinv){
  
  
  var_part_tr_wERSS += (as_scalar(sum(Vinv % S1))*xtx);
  
  
  neg_KL += (logbf + 0.5*(-2*as_scalar(sum((Vinv*trans(rbar_j)) % trans(x_j*trans(mu1))))+
    as_scalar(sum(Vinv % (S1+(mu1*trans(mu1)))))*xtx));
}

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List compute_ELBO_terms_rcpp (double var_part_tr_wERSS_init, double neg_KL_init, double x_j,
                              const mat& rbar_j, double logbf, const mat& mu1, const mat& S1, 
                              double xtx, const mat& Vinv) {
  
  double var_part_tr_wERSS = var_part_tr_wERSS_init;
  double neg_KL = neg_KL_init;
  
  compute_ELBO_terms(var_part_tr_wERSS, neg_KL, x_j, rbar_j, logbf, mu1, S1, xtx, Vinv);
  
  return List::create(Named("var_part_tr_wERSS") = var_part_tr_wERSS,
                      Named("neg_KL")            = neg_KL);
}


// Function to compute the variance part of the ERSS
void compute_var_part_ERSS (mat& var_part_ERSS, const mat& S1, double xtx);

void compute_var_part_ERSS (mat& var_part_ERSS, const mat& S1, double xtx){
  
  var_part_ERSS += (S1*xtx);
}

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
mat compute_var_part_ERSS_rcpp (mat& var_part_ERSS_init, const mat& S1, double xtx) {
  
  mat var_part_ERSS = var_part_ERSS_init;
  
  compute_var_part_ERSS(var_part_ERSS, S1, xtx);
  
  return var_part_ERSS;
}
