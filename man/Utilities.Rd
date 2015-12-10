\name{Utilities}
\alias{compute.acc}
\alias{compute.F}
\alias{norm1}
\alias{Unit.sphere.norm}
\alias{do.stratified.cv.data}
\alias{labelsfromscores}
\alias{Multiple.labels.from.scores}
\alias{selection.test}

\title{
 Utility functions
}

\description{
Mixed utility functions to compute accuracy, norms, labels from scores and to perform stratified cross-validation.
}

\usage{
compute.acc(pred, labels)
compute.F(pred, labels)
norm1(x)
Unit.sphere.norm(K)
do.stratified.cv.data(examples, positives, k = 5, seed = NULL)
labelsfromscores(scores, thresh)
Multiple.labels.from.scores(S, thresh.vect)
selection.test(pos.scores, av.scores, ind.positives, alpha = 0.05, thresh.pos = 0)
}

\arguments{
\item{pred}{vector of the predicted labels}

\item{labels}{vector of the true labels.
Note that 0  stands for negative and 1 for positive.
In general the first level is negative and the second positive}

\item{x}{numeric vector}

\item{K}{a kernel matrix}

\item{examples}{indices of the examples (a vector of integer)}

\item{positives}{vector of integer. Indices of the positive examples. The indices refer to the indices of examples}

\item{k}{number of folds (def = 5)}

\item{seed}{seed of the random generator (def=NULL). If is set to NULL no initiazitation is performed}

\item{scores}{numeric. Vector of scores: each element correspond to the score of an example}

\item{thresh}{real value. Threshold for the classification}

\item{S}{numeric matrix. Matrix of scores: rows represent examples, columns classes}

\item{thresh.vect}{numeric vector. Vector of the thresholds for multiple classes (one threshold for each class)}

\item{pos.scores}{vector with scores of positive examples. It is returned from multiple.ker.score.cv.}

\item{av.scores}{a vector with the average scores computed by multiple.ker.score.cv. It may be a named vector. 
If not, the names attributes corresponding to the indices of the vector are added.}

\item{ind.positives}{indices of the positive examples. They are the indices of av.scores corresponding to positive examples.}

\item{alpha}{quantile level (def. 0.05)}

\item{thresh.pos}{only values larger than thresh.pos are retained in pos.scores (def.: 0)}
}


\details{
\code{compute.acc} computes  the accuracy for a single class

\code{compute.F} computes  the F-score for a single class

\code{norm1} computes the L1-norm of a numeric vector

\code{Unit.sphere.norm} normalize a kernel according to the unit sphere

\code{do.stratified.cv.data} generates data for the stratified cross-validation. In particular subdividas the indices that refer to the rows of the data matrix in different folds (separated for positive and negative examples)

\code{labelsfromscores} computes the labels of a single class from the corresponding scores

\code{Multiple.labels.from.scores} computes the labels of multiple classes from the corresponding scores

\code{selection.test} is a non parametric test to select the most significant unlabeled examples 
}

\value{
\code{compute.acc} returns the accuracy

\code{compute.F} returns the F-score

\code{norm1} returns the L1-norm value

\code{Unit.sphere.norm} returns  the kernel normalized according to the unit sphere

\code{do.stratified.cv.data} returns a list with 2 two components:
\item{fold.non.positives}{a list with k components. Each component is a vector with the indices of the non positive elements of the fold}
\item{fold.positives}{a list with k components. Each component is a vector with the indices of the positive elements of the fold}
Indices refer to row numbers of the data matrix

\code{labelsfromscores} returns a numeric vector res with 0 or 1 values. The label res[i]=1 if scores[i]>thresh, otherwise res[i]=0

\code{Multiple.labels.from.scores} returns a binary matrix with the labels of the predictions. Rows represent examples, columns classes. Element L[i,j] is the label of example i w.r.t. class j.  L[i,j]=1 if i belongs to j, 0 otherwise.

\code{selection.test} returns a list with 5 components:
\item{selected}{a named vector with the components of av.scores selected by the test}
\item{selected.labeled}{a named vector with the labeled components of av.scores selected by the test}
\item{selected.unlabeled}{a named vector with the unlabeled components of av.scores selected by the test}
\item{thresh}{the score threshold selected by the test}
\item{alpha}{significance level (the same value of the input)}

}


\examples{
# L1-norm of a vector
norm1(rnorm(10));
# generation of 5 stratified folds;
do.stratified.cv.data(1:100, 1:10, k = 5, seed = NULL);
# generation of labels form scores.
labelsfromscores(runif(20), thresh=0.3);
}
