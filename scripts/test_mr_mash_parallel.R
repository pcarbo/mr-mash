suppressMessages(library(MBSP))
library(Rcpp)
library(RcppParallel)
source("../code/misc.R")
source("../code/bayes_mvr.R")
source("../code/mr_mash_simple.R")
sourceCpp("../code/mr_mash.cpp",verbose = TRUE)

# SCRIPT PARAMETERS
# -----------------
# Number of samples (n) and number of predictors (p).
n <- 200
p <- 250

# Residual covariance matrix.
V <- diag(100)
r <- nrow(V)

# True effects used to simulate the data.
B <- rbind(c(-2.0, -1.5),
           c( 1.0,  1.0),
           matrix(0,p - 2,r))

# Covariances in the mixture-of-normals prior on the regression
# coefficients.
k  <- 40
t  <- seq(0,1,length.out = k)
S0 <- vector("list",k)
names(S0) <- paste0("k",1:k)
for (i in 1:k)
  S0[[i]] <- t[i] + (1-t[i])*diag(r)
    
# The mixture weights in the mixture-of-normals prior on the
# regression coefficients.
w0 <- rep(1/k,k)
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
B0 <- matrix(0,p,r)
print(system.time(fit1 <- mr_mash_simple(X,Y,V,S0,w0,B0,20,version = "Rcpp")))

# Redo the computations using the multithreaded C++ implementation using TBB. We
# expect about a 4x speedup with 8 threads.
setThreadOptions(numThreads = 8)
print(system.time(fit2 <- mr_mash_simple(X,Y,V,S0,w0,B0,20,
                                         version = "RcppParallel")))

# Redo the computations using the multithreaded C++ implementation
# using OpenMP. We expect about a 4x speedup with 8 threads
# (set OMP_NUM_THREADS=8 before starting R).
print(system.time(fit3 <- mr_mash_simple(X,Y,V,S0,w0,B0,20,
                                         version = "RcppOpenMP")))

print(range(fit1$B - fit2$B))
print(range(fit1$B - fit3$B))
