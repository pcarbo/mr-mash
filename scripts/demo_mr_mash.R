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
B <- rbind(c(-2,-2),
           c(5,5),
           matrix(0,p - 2,R))

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
