# An illustration of the mr_mash_simple implementation applied to a
# small, simulated data set.
suppressMessages(library(MBSP))
library(mvtnorm)
library(Rcpp)
devtools::load_all("~/git/mr.mash.alpha")
source("../code/misc.R")
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

##################
####Centered X####
##################

###Center X
Xc <- scale(X, scale=F)

###Precompute quantities
comps_rcpp <-
  mr.mash.alpha:::precompute_quants(n=NULL, X=Xc, V=V, S0=S0,
                                    standardize=FALSE, version="Rcpp")
comps_r <-
    mr.mash.alpha:::precompute_quants(n=NULL, X=Xc, V=V, S0=S0,
                                      standardize=FALSE, version="R")

###Fit simple multivariate regression with mixture prior
out_rcpp <-
  bayes_mvr_mix_centered_X_rcpp(Xc[,1], Y, V, w0, simplify2array(S0),
                                comps_rcpp$xtx[1], comps_rcpp$V_chol,
                                comps_rcpp$U0, comps_rcpp$d, comps_rcpp$Q)
out_r <- mr.mash.alpha:::bayes_mvr_mix_centered_X(Xc[,1], Y, V, w0, S0,
           comps_r$xtx[1], comps_r$V_chol, comps_r$U0, comps_r$d, comps_r$Q)
print(range(drop(out_rcpp$mu1) - out_r$mu1))
print(range(out_rcpp$S1 - out_r$S1))
print(range(drop(out_rcpp$w1) - out_r$w1))
print(range(out_rcpp$logbf - out_r$logbf))

###Fit the inner loop
mu1  <- matrix(0, ncol=ncol(B), nrow=nrow(B))
rbar <- Y - Xc%*%mu1
out1_rcpp <- inner_loop_general_rcpp(X=Xc, rbar=rbar, mu1=mu1, V=V, w0=w0,
               S0=simplify2array(S0), precomp_quants=comps_rcpp,
               standardize=FALSE)
out1_r <- mr.mash.alpha:::inner_loop_general(X=Xc, rbar=rbar, mu=mu1, V=V,
            Vinv=NULL, w0=w0, S0=S0, precomp_quants=comps_r,
            standardize=FALSE, update_V=FALSE)
print(range(out1_rcpp$mu1 - out1_r$mu1))
print(range(out1_rcpp$S1 - out1_r$S1))
print(range(out1_rcpp$w1-out1_r$w1))
print(range(out1_rcpp$rbar-out1_r$rbar))

stop()

#################
####Scaled  X####
#################

###Scale X
Xs <- scale(X)

###Precompute quantities
comps_scaled_rcpp <- mr.mash.alpha:::precompute_quants(n=n, X=NULL, V=V, S0=S0, standardize=TRUE, version="Rcpp")
comps_scaled_r <- mr.mash.alpha:::precompute_quants(n=n, X=NULL, V=V, S0=S0, standardize=TRUE, version="R")

###Fit simple multivariate regression with mixture prior
out2_rcpp <- bayes_mvr_mix_scaled_X_rcpp(Xs[,1], Y, w0, simplify2array(S0), comps_scaled_rcpp$S, comps_scaled_rcpp$S1, 
                                         comps_scaled_rcpp$SplusS0_chol, comps_scaled_rcpp$S_chol, comps_scaled_rcpp$ldetSplusS0_chol, 
                                         comps_scaled_rcpp$ldetS_chol)
out2_r <- mr.mash.alpha:::bayes_mvr_mix_scaled_X(Xs[,1], Y, w0, S0, comps_scaled_r$S, comps_scaled_r$S1, comps_scaled_r$SplusS0_chol, comps_scaled_r$S_chol, 
                                                 comps_scaled_r$ldetSplusS0_chol, comps_scaled_r$ldetS_chol)
print(drop(out2_rcpp$mu1)-out2_r$mu1, 16)
print(out2_rcpp$S1-out2_r$S1, 16)
print(drop(out2_rcpp$w1)-out2_r$w1, 16)
print(out2_rcpp$logbf-out2_r$logbf, 16)
##These are close to 0

###Fit the inner loop
mu1 <- matrix(0, ncol=ncol(B), nrow=nrow(B))
rbar_s <- Y - Xs%*%mu1
out3_rcpp <- inner_loop_general_rcpp(X=Xs, rbar=rbar_s, mu1=mu1, V=V, w0=w0, S0=simplify2array(S0), precomp_quants=comps_scaled_rcpp, standardize=TRUE)
out3_r <- mr.mash.alpha:::inner_loop_general(X=Xs, rbar=rbar_s, mu=mu1, V=V, Vinv=NULL, w0=w0, S0=S0, precomp_quants=comps_scaled_r, standardize=TRUE, update_V=FALSE)
print(out3_rcpp$mu1-out3_r$mu1, 16)
print(out3_rcpp$S1-out3_r$S1, 16)
print(out3_rcpp$w1-out3_r$w1, 16)
print(out3_rcpp$rbar-out3_r$rbar, 16)
##These are close to 0
