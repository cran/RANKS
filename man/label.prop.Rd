\name{label.prop}
\alias{label.prop}

\title{
Label propagation
}


\description{
Function that implements the Label propagation algorithm of Zhu and Ghahramani
}

\usage{
label.prop(W, ind.positives, tmax = 1000, eps = 1e-05, norm = TRUE)
}

\arguments{
\item{W}{a numeric matrix representing the adjacency matrix of the graph}

\item{ind.positives}{indices of the "core" positive examples of the graph.
They represent the indices of W corresponding to the positive examples}

\item{tmax}{maximum number of iterations (def: 1000)}

\item{eps}{numeric. Maximum allowed difference between the computed probabilities at the steady state (def. 1e-5)}

\item{norm}{boolean. If TRUE (def) the adjacency matrix \eqn{W} of the graph is normalized to \eqn{M = D^{-1} * W}, otherwise
it is assumed that the matrix \eqn{W} is just normalized}
}

\details{
label.prop implements the label propagation algorithm on a given graph by performing 1 or more steps on the graph, depending on the value of the tmax parameter. It stops also if the difference of the norm of the scores between two consecutive steps is less than eps.
}

\value{
A list with three elements:
\item{p}{numeric vector. Scores of each node at the steady state or after tmax iterations}
\item{ind.positives}{indices of the "core" positive examples of the graph (it is equal to the same input parameter)}
\item{n.iter}{number of performed steps/iterations}
}

\references{
Zhu, X., Ghahramani, Z., Lafferty, J.: Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. In: Proc. of the Twentieth International Conference on Machine Learning, Washington DC (2003) 912-919
}



\examples{
# Application of label prop algorithm to the prediction of the DrugBank category Penicillins
# using the Tanimoto chemical structure similarity network 
# between 1253 DrugBank drugs
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
label.prop(DD.chem.data, ind.pos, tmax = 10, eps = 1e-05, norm = TRUE);
}
