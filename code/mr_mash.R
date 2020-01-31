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

# Bayesian multivariate regression with mixture-of-normals prior
# (mixture weights w0 and covariance matrices S0) 
#
# The outputs are: the log-Bayes factor (logbf), the posterior
# assignment probabilities (w1), the posterior mean of the
# coefficients given that all the coefficients are not nonzero (mu1),
# and the posterior covariance of the coefficients given that all the
# coefficients are not zero (S1).
bayes_mvr_mix <- function (x, Y, V, w0, S0) {
  
  # Get the dimension of the response (r) and the number of mixture
  # components (k).
  r <- ncol(Y)
  k <- length(w0)
  
  # Compute the quantities separately for each mixture component.
  out <- vector("list",k)
  for (i in 1:k)
    out[[i]] <- bayes_mvr_ridge(x,Y,V,S0[[i]])

  # Compute the posterior assignment probabilities for the latent
  # indicator variable.
  logbf <- sapply(out,"[[","logbf")
  w1    <- softmax(logbf + log(w0))

  # Compute the log-Bayes factor as a linear combination of the
  # individual Bayes factors for each mixture component.
  z     <- log(w0) + logbf
  u     <- max(z)
  logbf <- u + log(sum(exp(z - u)))
  
  # Compute the posterior mean (mu1) and covariance (S1) of the
  # regression coefficients.
  S1  <- matrix(0,r,r)
  mu1 <- rep(0,r)
  for (i in 1:k) {
    wi  <- w1[i]
    mui <- out[[i]]$mu1
    Si  <- out[[i]]$S1
    mu1 <- mu1 + wi*mui
    S1  <- S1 + wi*(Si + tcrossprod(mui))
  }
  S1 <- S1 - tcrossprod(mu1)
  
  # Return the the posterior mean (mu1) and covariance (S1), the
  # posterior assignment probabilities (w1), and the log-Bayes factor
  # (logbf).
  return(list(mu1   = mu1,
              S1    = S1,
              w1    = w1,
              logbf = logbf))
}
