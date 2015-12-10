\name{ker.score.classifier.holdout}
\alias{ker.score.classifier.holdout}

\title{
RANKS held-out procedure for a single class
}

\description{
Function to perform an held-out procedure for a single class with a kernel-based score method
}
\usage{
ker.score.classifier.holdout(K, ind.pos, ind.test, m = 5, p = 10, 
alpha = seq(from = 0.05, to = 0.6, by = 0.05), init.seed = 0, 
opt.fun = compute.F, fun = KNN.score, ...)
}

\arguments{
\item{K}{matrix. Kernel matrix or any valid symmetric matrix}

\item{ind.pos}{indices of the positive examples of the training set. They are the indices the row of RW corresponding to
positive examples of the training set}

\item{ind.test}{indices of the examples of the test set. They are the indices the row of RW corresponding to
examples of the test set}

\item{m}{number of folds for the cross-validation on the training set}

\item{p}{number of repeated  cross-validations on the training set}

\item{alpha}{vector of the quantiles to be tested}

\item{init.seed}{nitial seed for the random generator (def: 0)}

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
Function to classify labels according to an hold-out procedure with a kernel-based score method.
The optimal threshold for a given class is obtained by (possibly multiple) internal cross-validation on the training set.
Scores of the held-out nodes are computed. Thresholds are computed on the training set
by cross-validation and then are used to classify the held-out nodes in the test set.
The optimal quantile and corresponding threshold  are selected by internal cross-validation using the F-score as metrics.
Note the test examples are given as indices of the rows of the input matrix RW.
}

\value{
a list with four components:
A list with 4 components:
\item{labels}{vector of the predicted labels for the test set(1 represent positive, 0 negative)}
\item{av.scores }{a vector with the  scores computed on the test set.
              Elements of the vector av.scores correspond to ind.test rows of RW}
\item{opt.alpha }{the optimal quantile alpha}
\item{opt.thresh }{the optimal threshold}
}


\seealso{
\code{\link{rw.kernel-methods}}, \code{\link{Kernel functions}}, \code{\link{ker.score.classifier.cv}}
}
\examples{
# Node label classification of the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# with eav-score with 1-step random walk kernel
# using held-out with 5-fold CV repeated 10 times on the training set 
# to set the "optimal" threshold for classifiaction
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.test <- 1:300;
ind.train <- 301:length(labels);
ind.pos <- which(labels==1);
ind.pos <- ind.pos[ind.pos>300];
K <- rw.kernel(DD.chem.data);
res <- ker.score.classifier.holdout(K, ind.pos, ind.test, m = 5, p = 10, fun = eav.score);
}

