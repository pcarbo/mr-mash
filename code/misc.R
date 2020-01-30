# Return the dot product of vectors x and y.
dot <- function (x,y)
  sum(x*y)

# Return the quadratic norm (2-norm) of vector x.
norm2 <- function (x)
  sqrt(dot(x,x))
