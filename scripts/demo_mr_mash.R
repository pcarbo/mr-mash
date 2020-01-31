# TO DO: Explain here what this script does, and how to use it.
library(MBSP)
source("../code/misc.R")
source("../code/mr_mash.R")

# SCRIPT PARAMETERS
# -----------------
n <- 500
p <- 20
R <- 2

# The residual covariance matrix.
V <- rbind(c(1.0,0.2),
           c(0.2,0.4))
R <- nrow(V)

# The true effects used to simulate the data.
B <- rbind(c(-2.0, -1.5),
           c( 1.0,  1.0),
           matrix(0,p - 2,R))

# The covariances in the mixture-of-normals prior on the regression
# coefficients.
S0 <- list(k1 = rbind(c(4,0),
                      c(0,4)),
           k2 = rbind(c(4,2),
                      c(2,4)),
           k3 = rbind(c(4,3.5),
                      c(3.5,4)))

# The mixture weights in the mixture-of-normals prior on the
# regression coefficients.
w <- c(0.1,0.6,0.3)

# SIMULATE DATA
# -------------
set.seed(1)
X <- matrix(rnorm(n*p),n,p)
X <- scale(X,scale = FALSE)

# Simulate Y ~ MN(X*B,I,V). Note that matrix.normal from the MBSP
# package appears to be much faster than rmatrixnorm from the
# MixMatrix package.
Y <- matrix.normal(X %*% B,diag(n),V)

# FIT MR-MASH MODEL
# -----------------
# TO DO.
