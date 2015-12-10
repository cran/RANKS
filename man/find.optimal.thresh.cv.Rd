\name{find.optimal.thresh.cv}
\alias{find.optimal.thresh.cv}

\title{
Function to find the optimal RANKS score thereshold
}

\description{
Function to find the optimal quantile alpha and corresponding threshold by  cross-validation with a kernel-based
score method.
}

\usage{
find.optimal.thresh.cv(K, ind.pos, ind.non.pos, m = 5, 
alpha = seq(from = 0.05, to = 0.6, by = 0.05), init.seed = NULL, 
opt.fun = compute.F, fun = KNN.score, ...)
}

\arguments{
\item{K}{matrix. Kernel matrix or any valid symmetric matrix}

\item{ind.pos}{indices of the positive examples. They are the indices the row of RW corresponding to positive examples of the training set.}

\item{ind.non.pos}{indices of the non positive examples. They are the indices the row of RW corresponding to non positive examples  of the training set.}

\item{m}{number of folds (default: 5)}

\item{alpha}{vector of the quantiles to be tested}

\item{init.seed}{initial seed for the random generator. If NULL (def) no initialization is performed}

\item{opt.fun}{Function implementing the metric to select the optimal threshold.
The F-score (compute.F) is the default. Available functions:

- compute.F: F-score (default) 
 
- compute.acc:accuracy.  

Any function having two arguments representing the vector of predicted and true labels can be in principle used.
}

\item{fun}{function. It must be a kernel-based score method (default KNN.score)}

\item{...}{optional arguments for the function fun}

}

\details{
Function to find the optimal quantile alpha and corresponding threshold by  cross-validation with a kernel-based
score method. The optimality is computed with respect to a specific metric (def: F-score).
This function is used by \code{multiple.ker.score.thresh.cv}, \code{ker.score.classifier.holdout}, \code{ker.score.classifier.cv}.
}
\value{
A list with 3 elements:
\item{alpha}{quantile corresponding to the best F-score}
\item{thresh}{threshold corresponding to the best F-score}
\item{pos.scores}{scores of the positive elements computed through CV}
}

\seealso{
\code{\link{multiple.ker.score.thresh.cv}}, \code{\link{Kernel functions}}, \code{\link{ker.score.classifier.holdout}}
}

\examples{
# Finding the optimal threshold in the Tanimoto chemical structure similarity network 
# between 1253 DrugBank drugs for the prediction of the DrugBank category Penicillins using
# the KNN-score with the random walk kernel 
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
K <- rw.kernel(DD.chem.data);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
ind.non.pos <- which(labels==0);
res <- find.optimal.thresh.cv(K, ind.pos, ind.non.pos);
res
}

