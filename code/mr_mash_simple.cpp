// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

// FUNCTION DEFINITIONS
// --------------------
//
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double hello_world_rcpp (const arma::vec& x) {
  double y = sum(x);
  return y;
}
