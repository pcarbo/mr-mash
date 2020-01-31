# TO DO: Explain here what this script does, and how to use it.
suppressMessages(library(MBSP))
library(mvtnorm)
source("../code/misc.R")
source("../code/mr_mash.R")

# SCRIPT PARAMETERS
# -----------------
n <- 500
p <- 20
r <- 2

# The residual covariance matrix.
V <- rbind(c(1.0,0.2),
           c(0.2,0.4))
r <- nrow(V)

# The true effects used to simulate the data.
B <- rbind(c(-2.0, -1.5),
           c( 1.0,  1.0),
           matrix(0,p - 2,r))

# The covariances in the mixture-of-normals prior on the regression
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
B0 <- matrix(0,p,r)
B1 <- mr_mash_update(X,Y,B,V,w0,S0)

# Test univariate computations:
s0      <- lapply(S0,"[",1)
s0[[1]] <- 1e-10
b1      <- mr_mash_update(X,Y[,1],B0[,1],V[1],w0,s0)

s0    <- unlist(s0)
s0[1] <- 0
out   <- varbvsmix(X,NULL,Y[,1],V[1]*s0,V[1],w0,matrix(0,p,k),matrix(0,p,k),
                   update.sigma = FALSE,update.sa = FALSE, update.w = FALSE,
                   maxiter = 1,drop.threshold = 0,verbose = FALSE)
b2    <- rowSums(out$alpha * out$mu)
print(range(b1 - b2))

# Test univariate computations:
# out <- bayes_mvr_mix(X[,3],Y[,1],V[1],w0,lapply(S0,"[",1))

# Test computation of quantities for basic multivariate regression model.
# out1 <- bayes_mvr_mix(X[,3],Y,V,w0,S0)
# source("~/git/mr.mash.alpha/R/bayes_reg_mv.R")
# out2 <- bayes_mvr_mix(X[,3],Y,V,w0,S0)

