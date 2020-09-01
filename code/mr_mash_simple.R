# Run several iterations of the co-ordinate ascent updates for the
# mr-mash model.
#
# The special case of univariate regression, when Y is a vector or a
# matrix with 1 column, is also handled.
#
# This implementation is meant to be "instructive"---that is, I've
# tried to make the code as simple as possible, with an emphasis on
# clarity. Very little effort has been devoted to making the
# implementation efficient, or the code concise.
mr_mash_simple <- function (X, Y, V, S0, w0, B, numiter = 100,
                            version = c("R","Rcpp","RcppParallel", "RcppOpenMP")) {
  version <- match.arg(version)
  Y       <- as.matrix(Y)
  r       <- ncol(Y)
  k       <- length(w0)
  
  # This variable is used to keep track of the algorithm's progress.
  maxd <- rep(0,numiter)

  # Iterate the updates.
  for (i in 1:numiter) {

    # Save the current estimates of the posterior means.
    B0 <- B
      
    # Update the posterior means of the regression coefficients.
    if (version == "R")
      B <- mr_mash_update_simple(X,Y,B,V,w0,S0)
    else if (version == "Rcpp")
      B <- drop(mr_mash_update_rcpp(X,Y,as.matrix(B),as.matrix(V),w0,
                                    array(simplify2array(S0),c(r,r,k)),
                                    parallell = 'no'))
    else if (version == "RcppParallel")
      B <- drop(mr_mash_update_rcpp(X,Y,as.matrix(B),as.matrix(V),w0,
                                    array(simplify2array(S0),c(r,r,k)),
                                    parallell = 'TBB'))
    else if (version == "RcppOpenMP")
      B <- drop(mr_mash_update_rcpp(X,Y,as.matrix(B),as.matrix(V),w0,
                                    array(simplify2array(S0),c(r,r,k)),
                                    parallell = 'OpenMP'))
    
    # Store the largest change in the posterior means.
    maxd[i] <- abs(max(B - B0))
  }

  # Return the updated posterior means of the regression coefficicents
  # (B) and the maximum change at each iteration (maxd).
  return(list(B = B,maxd = maxd))
}

# Perform a single pass of the co-ordinate ascent updates for the
# mr-mash model.
#
# The special case of univariate regression, when Y is a vector or a
# matrix with 1 column, is also handled.
#
# This implementation is meant to be "instructive"---that is, I've
# tried to make the code as simple as possible, with an emphasis on
# clarity. Very little effort has been devoted to making the
# implementation efficient, or the code concise.
mr_mash_update_simple <- function (X, Y, B, V, w0, S0) {
    
  # Make sure B is a matrix.
  B <- as.matrix(B)
    
  # Get the number of predictors.
  p <- ncol(X)

  # Compute the expected residuals.
  R <- Y - X %*% B

  # Repeat for each predictor.
  for (i in 1:p) {
    x <- X[,i]
    b <- B[i,]
    
    # Disregard the ith predictor in the expected residuals.
    R <- R + outer(x,b)

    # Update the posterior of the regression coefficients for the ith
    # predictor.
    out   <- bayes_mvr_mix_simple(x,R,V,w0,S0)
    b     <- out$mu1
    B[i,] <- b
    
    # Update the expected residuals.
    R <- R - outer(x,b)
  }

  # Output the updated posterior mean coefficients.
  return(drop(B))
}
