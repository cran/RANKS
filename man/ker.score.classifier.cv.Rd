\name{ker.score.classifier.cv}
\alias{ker.score.classifier.cv}

\title{
Multiple cross-validation with RANKS for classification
}
\description{
Function to classify labels according to an external cross-validation procedure with a kernel-based score method.
}
\usage{
ker.score.classifier.cv(K, ind.pos, m = 5, p = 100, 
alpha = seq(from = 0.05, to = 0.6, by = 0.05), init.seed = 0, 
opt.fun = compute.F, fun = KNN.score, ...)
}

\arguments{
\item{K}{matrix. Kernel matrix or any valid symmetric matrix}

\item{ind.pos}{indices of the positive examples. They are the row indices  of RW corresponding to positive examples.}

\item{m}{number of folds for each cross-validation}

\item{p}{number of repeated cross-validations}

\item{alpha}{vector of the quantiles to be tested}

\item{init.seed}{initial seed for the random generator (def: 0)}

\item{opt.fun}{: function. Function implementing the metric to choice the optimal threshold.
The F-score (compute.F) is the default. Available functions:

- compute.F: F-score (default) 
 
- compute.acc: accuracy.  
 
Any function having two arguments representing the vector of predicted and true labels can be in principle used.
}

\item{fun}{function. It must be a kernel-based score method (default KNN.score)}

\item{...}{optional arguments for the function fun}
}


\details{
Function to classify labels according to an external cross-validation procedure with a kernel-based score method. The optimal
threshold for a given class id found by internal cross-validation.
Scores are computed by averaging across (possibly) multiple external cross-validations.
The optimal quantile and corresponding threshold  are selected by internal cross-validation using the F-score (default) or the accuracy as metric.
}

\value{
A list with 4 components:
\item{labels}{vector of the predicted labels (1 represents positive, 0 negative)}
\item{av.scores }{a vector with the average scores across multiple cross-validations.
              Elements of the vector av.scores correspond to the rows of RW}
\item{opt.alpha }{the optimal quantile alpha}
\item{opt.thresh }{the optimal threshold}

a vector of the predicted scores for the test set
}


\seealso{
\code{\link{rw.kernel-methods}}, \code{\link{Kernel functions}}, \code{\link{ker.score.classifier.holdout}}
}
\examples{
# Nodel label classification of the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation repeated 3 times 
# and NN-score with 1-step random walk kernel
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
K <- rw.kernel(DD.chem.data);
res <- ker.score.classifier.cv(K, ind.pos, m = 5, p = 3, fun = NN.score);
}

