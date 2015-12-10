\name{multiple.ker.score.cv}
\alias{multiple.ker.score.cv}

\title{
RANKS multiple cross-validation for a single class 
}
\description{
Function to execute multiple cross-validation with RANKS for a single class.
}
\usage{
multiple.ker.score.cv(RW, ind.pos, m = 5, p = 100, init.seed = 0, fun = KNN.score, ...)
}

\arguments{
\item{RW}{matrix. Kernel matrix or any valid symmetric matrix}

\item{ind.pos}{indices of the positive examples. They are the row indices  of RW corresponding to positive examples.}

\item{m}{number of folds for each cross-validation}

\item{p}{number of repeated cross-validations}

\item{init.seed}{initial seed for the random generator (def: 0)}

\item{fun}{function. It must be a kernel-based score method (default KNN.score)}

\item{...}{optional arguments for the function fun}
}


\details{
It performs multiple cross-validation using RANKS to predict the cross-validated scores. The cross-validation is stratified: 
the folds are constructed separately for each class, to maintain an equal ratio between classes among folds.
It computes the scores by averaging across multiple cross validations.
}
\value{
A list with two components:
\item{av.scores }{a vector with the average scores across multiple cross-validations.
              Elements of the vector av.scores correspond to the rows of RW}
\item{pos.scores }{a vector with the scores of positive elements collected at each iteration}

}


\seealso{
\code{\link{ker.score.cv}}, \code{\link{multiple.ker.score.thresh.cv}}, \code{\link{rw.kernel-methods}}, \code{\link{Kernel functions}}.
}
\examples{
# Nodel label ranking for the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation repeated 10 times
# and eav-score with 1-step random walk kernel
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
K <- rw.kernel(DD.chem.data);
res <- multiple.ker.score.cv(K, ind.pos, m = 5, p = 10, init.seed = 0, fun = eav.score);
# the same but using the NN-score
res <- multiple.ker.score.cv(K, ind.pos, m = 5, p = 10, init.seed = 0, fun = NN.score);
}

