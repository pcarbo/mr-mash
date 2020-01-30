# Bayesian multivariate regression with Normal prior. The outputs are:
# b, the least-squares estimate of the regression coefficients; S, the
# covariance of b; mu1, the posterior mean of the regression
# coefficients; S1, the posterior covariance of the regression
# coefficients; logbf, the log-Bayes factor.
bayes_mvr_ridge <- function (x, Y, V, S0) {
  
  # Compute the least-squares estimate of the coefficients (b) and their
  # covariance (S).
  xx   <- norm2(x)^2
  bhat <- drop(x %*% Y)/xx
  S    <- V/xx
  
  # Compute the log-Bayes factor.
  logbf <- (as.numeric(determinant(S)$modulus) +
              - as.numeric(determinant(S0 + S)$modulus)
            + dot(bhat,solve(S,bhat)) - dot(bhat,solve(S0 + S,bhat)))/2
  
  # Compute the posterior mean and covariance assuming a multivariate
  # normal prior with zero mean and covariance S0.
  SplusS0_chol <- chol(S+S0)
  S1 <- S0%*%backsolve(SplusS0_chol, forwardsolve(t(SplusS0_chol), S))
  S_chol <- chol(S)
  mu1    <- drop(S1%*%backsolve(S_chol, forwardsolve(t(S_chol), b)))
  
  # Return the least-squares estimate and covariance (b, S), the
  # posterior mean and covariance (mu1, S1), and the log-Bayes factor
  # (logbf)
  return(list(b = b,S = S,mu1 = mu1,S1 = S1,logbf = logbf))
}
