\name{ker.score.cv}
\alias{ker.score.cv}

\title{
RANKS cross-validation for a single class
}
\description{
Function to perform cross-validation for a single class with a kernel-based score method
}
\usage{
ker.score.cv(RW, ind.pos, m = 5, init.seed = NULL, fun = KNN.score, ...)
}

\arguments{
 \item{RW}{matrix. It can be a kernel matrix or the adjacency matrix of a graph}

\item{ind.pos}{indices of the positive examples. They are the row indices  of RW corresponding to positive examples.}

\item{m}{number of folds (def: 5)}

\item{init.seed}{initial seed for the random generator to generate folds. If NULL (default) no initialization is performed}

\item{fun}{function. It must be a kernel-based score method (default KNN.score)}

\item{...}{optional arguments for the function fun}
}

\details{
It performs a cross-validation using RANKS to predict the cross-validated scores. The cross-validation is stratified: 
the folds are constructed separately for each class, to maintain an equal ratio between classes among folds.
}

\value{
a numeric vector with the scores computed for each example
}


\seealso{
\code{\link{multiple.ker.score.cv}}, \code{\link{multiple.ker.score.thresh.cv}}, \code{\link{rw.kernel-methods}}, \code{\link{Kernel functions}}.
}
\examples{
# Nodel label ranking of the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation
# and eav-score with 1-step random walk kernel
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
K <- rw.kernel(DD.chem.data);
res <- ker.score.cv(K, ind.pos, m = 5, init.seed = NULL, fun = eav.score);
}
