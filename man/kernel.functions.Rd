\name{kernel.functions}
\alias{Kernel functions}
\alias{cauchy.kernel}
\alias{laplacian.kernel}
\alias{gaussian.kernel}
\alias{inv.multiquadric.kernel}
\alias{identity.kernel}
\alias{linear.kernel}
\alias{poly.kernel}
\title{
Kernel functions
}
\description{
Compute similarities between feature vectors according to a specific kernel function
}
\usage{
cauchy.kernel(W, sigma = 1)
laplacian.kernel(W, sigma = 1)
gaussian.kernel(W, sigma = 1)
inv.multiquadric.kernel(W, v = 1)
identity.kernel(W, a = 1)
linear.kernel(W, a = 1)
poly.kernel(W, degree = 2, scale = -1, v = 0)
}

\arguments{
  \item{W}{
a numeric matrix, Rows are examples and columns are features
}
  \item{sigma}{
a real value representing the sigma parameter (def. 1) of the Cauchy, Gaussian and Laplacian kernel
}

\item{v}{
constant factor (def. 1) of the inverse multiquadric kernel and of the polynomail kernel; for the inverse multiquadric kernel v must be larger than 0.
}

\item{a}{
unused parameter, maintained for compatibility reasons .
}

  \item{degree}{
integer corresponding to a degree of the polynomial (def. 2)
}
  \item{scale}{
double: scaling factor of the polynomial kernel. If \eqn{scale=-1} (def) scale is set to \eqn{1/ncol(W)};
}


   
}
\details{
All the kernel matrices are computed by calling C code to speed-up the computation.

\code{cauchy.kernel} computes the Cauchy kernel.

\code{laplacian.kernel} computes the Lapalacian kernel.

\code{gaussian.kernel} computes the Gaussian kernel.

\code{inv.multiquadric.kernel} computes the inverse multiquadric kernel.

\code{identity.kernel} computes the identity kernel. In this case the input W  represents a similarity square matrix (obtained i.e. through the Pearson correlation) between examples.

\code{linear.kernel} computes the linear kernel.




}
\value{
A kernel matrix representing the similarities between the examples (rows of W), according to a specific kernel function. 
}


\seealso{
\code{\link{rw.kernel-methods}}
}
\examples{
# computing kernels on the Tanimoto chemical structure similarity matrix
library(bionetdata);
data(DD.chem.data);
K <- identity.kernel(DD.chem.data);
K <- linear.kernel(DD.chem.data);
\dontrun{
K <- gaussian.kernel(DD.chem.data);
K <- inv.multiquadric.kernel(DD.chem.data);
K <- poly.kernel(DD.chem.data);
}
}
