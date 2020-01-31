# Compute quantities for a basic Bayesian multivariate regression with
# a multivariate normal prior on the regression coefficients: Y = xb'
# + E, E ~ MN(0,I,V),b ~ N(0,S0). The outputs are: bhat, the
# least-squares estimate of the regression coefficients; S, the
# covariance of bhat; mu1, the posterior mean of the regression
# coefficients; S1, the posterior covariance of the regression
# coefficients; and logbf, the logarithm of the Bayes factor.
#
# The special case of univariate regression is also handled.
#
bayes_mvr_ridge <- function (x, Y, V, S0) {

  # Make sure that Y, V and S0 are matrices.
  Y  <- as.matrix(Y)
  V  <- as.matrix(V)
  S0 <- as.matrix(S0)
    
  # Compute the least-squares estimate of the coefficients (bhat) and
  # the covariance of the standard error (S).
  xx   <- norm2(x)^2
  bhat <- drop(x %*% Y)/xx
  S    <- V/xx
  
  # Compute the posterior mean (mu1) and covariance (S1) assuming a
  # multivariate normal prior with zero mean and covariance S0.
  r   <- ncol(Y)
  I   <- diag(r)
  S1  <- S0 %*% solve(I + solve(S) %*% S0)
  mu1 <- drop(S1 %*% solve(S,bhat))

  # Compute the log-Bayes factor.
  logbf <- ldmvnorm(bhat,S0 + S) - ldmvnorm(bhat,S)
  
  # Return the least-squares estimate (bhat) and its covariance (S), the
  # posterior mean (mu1) and covariance (S1), and the log-Bayes factor
  # (logbf).
  return(list(bhat  = bhat,
              S     = drop(S),
              mu1   = mu1,
              S1    = drop(S1),
              logbf = logbf))
}
