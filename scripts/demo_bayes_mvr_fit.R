# TO DO: Explain here what this script does, and how to use it.
suppressMessages(library(MBSP))
library(mvtnorm)
library(Rcpp)
source("../code/misc.R")
source("../code/bayes_mvr.R")

# SCRIPT PARAMETERS
# -----------------
# Number of samples.
n <- 500

# Residual covariance matrix.
V <- rbind(c(1.0,0.2),
           c(0.2,0.4))
r <- nrow(V)

# True effects used to simulate the data.
b <- c(-2,0)

# Normal prior on the regression
# coefficient.
S0 <- rbind(c(4,2),
            c(2,4))

# The mixture weights in the mixture-of-normals prior on the
# regression coefficients.
w0 <- c(0.1,0.6,0.2,0.1)
k  <- length(w0)

# SIMULATE DATA
# -------------
set.seed(1)
x <- rnorm(n)
x <- x - mean(x)

# Simulate Y ~ MN(x*b',I,V). Note that matrix.normal from the MBSP
# package appears to be much faster than rmatrixnorm from the
# MixMatrix package.
Y <- matrix.normal(outer(x,b),diag(n),V)
Y <- scale(Y,scale = FALSE)

# FIT PRIOR VARIANCE (s0)
# -----------------------
fit <- bayes_mvr_ridge_fit(x,Y,V,S0,numiter = 10)
    
# TO DO: Test bayes_mvr_ridge_fit in univariate case (when r = 1).
