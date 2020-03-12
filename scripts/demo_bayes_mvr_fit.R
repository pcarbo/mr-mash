# This script illustrates the use of EM to compute the
# maximum-likelihood estimate (MLE) of the prior variance (sigma0) in
# Bayesian multivariate regression.
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

# Covariances in the mixture-of-normals prior on the regression
# coefficients. The first covariance matrix is used for the basic
# multivariate regression.
S0 <- list(k1 = rbind(c(4,2),
                      c(2,4)),
           k2 = rbind(c(3,0),
                      c(0,3)),
           k3 = rbind(c(6,3.5),
                      c(3.5,4)),
           k4 = rbind(c(5,0),
                      c(0,0.1)))

# The mixture weights in the mixture-of-normals prior on the
# regression coefficients.
w0 <- c(0.1,0.6,0.2,0.1)
k  <- length(w0)

# SIMULATE DATA
# -------------
set.seed(4)
x <- rnorm(n)
x <- x - mean(x)

# Simulate Y ~ MN(x*b',I,V).
Y <- matrix.normal(outer(x,b),diag(n),V)
Y <- scale(Y,scale = FALSE)

# FIT BASIC MULTIVARIATE REGRESSION MODEL
# ---------------------------------------
# Compute the maximum-likelihood estimate (MLE) of the prior variance
# (sigma0) for the basic multivariate regression model.
fit1 <- bayes_mvr_ridge_fit(x,Y,V,S0$k1,numiter = 5)
plot(1:5,max(fit1$logbf) - fit1$logbf + 1e-8,type = "l",log = "y",
     col = "dodgerblue",lwd = 2,xlab = "iteration",
     ylab = "distance to best logBF")

# FIT MODEL WITH MIXTURE PRIOR
# ----------------------------
# Compute the maximum-likelihood estimate (MLE) of the prior variance
# (sigma0) for the multivariate regression model with a
# mixture-of-normals prior on the regression coefficients.
fit2 <- bayes_mvr_mix_fit(x,Y,V,w0,S0,numiter = 8)
plot(1:8,max(fit2$logbf) - fit2$logbf + 1e-8,type = "l",log = "y",
     col = "magenta",lwd = 2,xlab = "iteration",
     ylab = "distance to best logBF")

fit2 <- bayes_mvr_mix_fit(x,Y[,1],V[1],w0,lapply(S0,function (x) x[1,1]),
                          numiter = 8)
