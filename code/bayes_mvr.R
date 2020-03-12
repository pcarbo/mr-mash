# Compute quantities for a basic Bayesian multivariate regression with
# a multivariate normal prior on the regression coefficients: Y = xb'
# + E, E ~ MN(0,I,V), b ~ N(0,S0). The outputs are: bhat, the
# least-squares estimate of the regression coefficients; S, the
# covariance of bhat; mu1, the posterior mean of the regression
# coefficients; S1, the posterior covariance of the regression
# coefficients; and logbf, the logarithm of the Bayes factor.
#
# The special case of univariate regression, when Y is a vector or a
# matrix with 1 column, is also handled.
#
# This implementation is meant to be "instructive"---that is, I've
# tried to make the code as simple as possible, with an emphasis on
# clarity. Very little effort has been devoted to making the
# implementation efficient, or the code concise.
bayes_mvr_ridge_simple <- function (x, Y, V, S0) {

  # Make sure Y, V and S0 are matrices.
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

# Compute quantities for Bayesian multivariate regression with a
# mixture-of-multivariate-normals prior on the regression
# coefficients. mu1, the posterior mean of the regression
# coefficients; S1, the posterior covariance of the regression
# coefficients; w1, the posterior "weights" for the individual
# components; and logbf, the logarithm of the Bayes factor.
#
# The special case of univariate regression, when Y is a vector or a
# matrix with 1 column, is also handled.
#
# This implementation is meant to be "instructive"---that is, I've
# tried to make the code as simple as possible, with an emphasis on
# clarity. Very little effort has been devoted to making the
# implementation efficient, or the code concise.
bayes_mvr_mix_simple <- function (x, Y, V, w0, S0) {
    
  # Make sure Y is a matrix.
  Y <- as.matrix(Y)
  
  # Get the dimension of the response (r) and the number of mixture
  # components (k).
  r <- ncol(Y)
  k <- length(w0)

  # Compute the quantities separately for each mixture component.
  out <- vector("list",k)
  for (i in 1:k)
    out[[i]] <- bayes_mvr_ridge_simple(x,Y,V,S0[[i]])
  
  # Compute the posterior assignment probabilities for the latent
  # indicator variable.
  logbf <- sapply(out,"[[","logbf")
  z     <- logbf + log(w0)
  w1    <- softmax(z)

  # Compute the log-Bayes factor as a linear combination of the
  # individual Bayes factors for each mixture component.
  u     <- max(z)
  logbf <- u + log(sum(exp(z - u)))
  
  # Compute the posterior mean (mu1) and covariance (S1) of the
  # regression coefficients.
  S1  <- matrix(0,r,r)
  mu1 <- rep(0,r)
  for (i in 1:k) {
    w   <- w1[i]
    mu  <- out[[i]]$mu1
    S   <- out[[i]]$S1
    mu1 <- mu1 + w*mu
    S1  <- S1 + w*(S + tcrossprod(mu))
  }
  S1 <- S1 - tcrossprod(mu1)
  
  # Return the the posterior mean (mu1) and covariance (S1), the
  # posterior assignment probabilities (w1), and the log-Bayes factor
  # (logbf).
  return(list(mu1   = mu1,
              S1    = drop(S1),
              w1    = w1,
              logbf = logbf))
}

# Compute a maximum-likelihood estimate (MLE) of the prior variance in
# the basic Bayesian multivariate regression model in which the prior
# covariance matrix is sigma0 * S0:
#
#   Y = xb' + E
#   E ~ MN(0,I,V)
#   b ~ N(0,sigma0*S0)
#
# A simple EM algorithm is used to compute the MLE. Note that this
# particular implementation requires that S0 be symmetric positive
# definite.
bayes_mvr_ridge_fit <- function (x, Y, V, S0, sigma0 = 1, numiter = 10) {

  # Record the Bayes factor at each EM iteration.
  logbf <- rep(0,numiter)
    
  # Iterate the EM updates.
  for (i in 1:numiter) {

    # Compute the log-Bayes factor and U = E[b'*P0*b], the second
    # moment of the regression coefficients, b, scaled by the prior
    # precision matrix, P0 (this is the inverse of the prior
    # covariance, S0). This is the E-step.
    out      <- bayes_mvr_ridge_simple(x,Y,V,sigma0*S0)
    mu1      <- out$mu1
    S1       <- out$S1
    U        <- solve(S0,S1 + tcrossprod(mu1))
    logbf[i] <- out$logbf

    # Compute the M-step update of the prior variance.
    sigma0 <- mean(diag(U))
  }

  # Return the model parameters (V, S0, sigma0) and the log-Bayes factor
  # at each EM iteration.
  return(list(V      = V,
              S0     = S0,
              sigma0 = sigma0,
              logbf  = logbf))
}

# Run several EM updates to compute a maximum-likelihood estimate
# (MLE) of the prior variance in a Bayesian multivariate regression
# with the regression coefficients assigned a simple mixture-of-normals
# prior. This is the same as bayes_mvr_ridge_fit, except that the
# normal prior is replaced with a mixture-of-normals prior.
bayes_mvr_mix_fit <- function (x, Y, V, w0, S0, sigma0 = 1, numiter = 10) {

  # Make sure Y is a matrix.
  Y  <- as.matrix(Y)

  # Get the dimension of the response (r) and the number of mixture
  # components (k).
  r <- ncol(Y)
  k <- length(w0)

  # Record the Bayes factor at each EM iteration.
  logbf <- rep(0,numiter)
    
  # Iterate the EM updates.
  for (i in 1:numiter) {

    # Compute the log-Bayes factor and U = E[b'*P0*b], the second
    # moment of the regression coefficients, b, scaled by the prior
    # precision matrix, P0 (this is the inverse of the prior
    # covariance, S0). This is the E-step.
    out <- bayes_mvr_mix_simple(x,Y,V,w0,lapply(S0,function (x) sigma0*x))
    logbf[i] <- out$logbf
    U        <- matrix(0,r,r)
    w1       <- out$w1
    for (i in 1:k) {
      out <- bayes_mvr_ridge_simple(x,Y,V,sigma0*S0[[i]])
      mu1 <- out$mu1
      S1  <- out$S1
      Ui  <- solve(S0[[i]],S1 + tcrossprod(mu1))
      U   <- U + w1[i]*Ui
    }
    
    # Compute the M-step update of the prior variance.
    sigma0 <- mean(diag(U))
  }

  # Return the model parameters (V, w0, S0, sigma0) and the log-Bayes
  # factor achieved at each EM iteration.
  return(list(V      = V,
              w0     = w0,
              S0     = S0,
              sigma0 = sigma0,
              logbf  = logbf))

}
