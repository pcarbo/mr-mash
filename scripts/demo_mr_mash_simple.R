# An illustration of the mr_mash_simple implementation applied to a
# small, simulated data set.
suppressMessages(library(MBSP))
library(mvtnorm)
library(Rcpp)
source("../code/misc.R")
source("../code/bayes_mvr.R")
source("../code/mr_mash_simple.R")
sourceCpp("../code/mr_mash.cpp",verbose = TRUE)

# SCRIPT PARAMETERS
# -----------------
# Number of samples (n) and number of predictors (p).
n <- 500
p <- 20

# Residual covariance matrix.
V <- rbind(c(1.0,0.2),
           c(0.2,0.4))
r <- nrow(V)

# True effects used to simulate the data.
B <- rbind(c(-2.0, -1.5),
           c( 1.0,  1.0),
           matrix(0,p - 2,r))

# Covariances in the mixture-of-normals prior on the regression
# coefficients.
S0 <- list(k1 = rbind(c(3,0),
                      c(0,3)),
           k2 = rbind(c(4,2),
                      c(2,4)),
           k3 = rbind(c(6,3.5),
                      c(3.5,4)),
           k4 = rbind(c(5,0),
                      c(0,0)))

# The mixture weights in the mixture-of-normals prior on the
# regression coefficients.
w0 <- c(0.1,0.6,0.2,0.1)
k  <- length(w0)

# SIMULATE DATA
# -------------
set.seed(1)
X <- matrix(rnorm(n*p),n,p)
X <- scale(X,scale = FALSE)

# Simulate Y ~ MN(X*B,I,V). Note that matrix.normal from the MBSP
# package appears to be much faster than rmatrixnorm from the
# MixMatrix package.
Y <- matrix.normal(X %*% B,diag(n),V)
Y <- scale(Y,scale = FALSE)

# FIT MR-MASH MODEL
# -----------------
# Run 20 co-ordinate ascent updates.
B0   <- matrix(0,p,r)
fit1 <- mr_mash_simple(X,Y,V,S0,w0,B0,20)

# Compare the posterior mean estimates of the regression coefficients
# against the coefficients used to simulate the data.
plot(B,fit1$B,pch = 20,xlab = "true",ylab = "estimated")
abline(a = 0,b = 1,col = "skyblue",lty = "dotted")

# Test the C++ code.
fit2 <- mr_mash_simple(X,Y,V,S0,w0,B0,20,version = "Rcpp")
print(range(fit1$B - fit2$B))
