\name{multiple.ker.score.thresh.cv}
\alias{multiple.ker.score.thresh.cv}

\title{
Function for RANKS multiple cross-validation and optimal threshold finding for a single class
}
\description{
Function to execute multiple cross-validation and to find the optimal threshold with RANKS for a single class.
}
\usage{
multiple.ker.score.thresh.cv(K, ind.pos, m = 5, p = 100, 
alpha = seq(from = 0.05, to = 0.6, by = 0.05), 
init.seed = 0, fun = KNN.score, ...)
}

\arguments{
\item{K}{matrix. Kernel matrix or any valid symmetric matrix}

\item{ind.pos}{indices of the positive examples. They are the row indices  of RW corresponding to positive examples.}

\item{m}{number of folds for each cross-validation}

\item{p}{number of repeated cross-validations}

\item{alpha}{vector of the quantiles to be tested}

\item{init.seed}{initial seed for the random generator (def: 0)}

\item{fun}{function. It must be a kernel-based score method  (default KNN.score)}

\item{...}{optional arguments for the function fun}
}

\details{
Function to execute multiple cross-validation with a kernel-based score method and to find the optimal
threshold for a given class by internal cross-validation.

Scores are computed by averaging across multiple external cross-validations.
The optimal quantile and corresponding threshold  are selected by internal cross-validation using a
specific metric (def: F-score).
}
\value{
A list with three components:
\item{av.scores }{a vector with the average scores across multiple cross-validations.
              Elements of the vector av.scores correspond to the rows of RW}
\item{opt.alpha }{the optimal quantile alpha}
\item{opt.thresh }{the optimal threshold}
}



\seealso{
\code{\link{ker.score.cv}}, \code{\link{multiple.ker.score.cv}}, \code{\link{rw.kernel-methods}}, \code{\link{Kernel functions}}.
}
\examples{
# Node label ranking and best threshold search for the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation repeated 2 times
# and eav-score with 1-step random walk kernel
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
K <- rw.kernel(DD.chem.data);
res <- multiple.ker.score.thresh.cv (K, ind.pos, m = 5, p = 2, init.seed = 0, fun = KNN.score);
}

