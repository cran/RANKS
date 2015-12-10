\name{RW}
\alias{RW}

\title{
Random walk on a graph
}

\description{
The function performs a random Walk on a given graph.
}

\usage{
RW(W, ind.positives, tmax = 1000, eps = 1e-10, norm = TRUE)
}


\arguments{
\item{W}{a numeric matrix representing the adjacency matrix of the graph}

\item{ind.positives}{indices of the "core" positive examples of the graph.
They represent the indices of W corresponding to the positive examples}

\item{tmax}{maximum number of iterations (steps) (def: 1000)}

\item{eps}{maximum allowed difference between the computed probabilities at the steady state (def. 1e-10)}

\item{norm}{if TRUE (def) the adjacency matrix \eqn{W} of the graph is normalized to \eqn{M = D^{-1} * W}, otherwise
it is assumed that the matrix \eqn{W} is just normalized}
}

\details{
RW performs a random Walk on a given graph by performing 1 or more steps on the graph, depending on the value of the tmax parameter.
It stops also if the difference of the norm of the probabilities between two consecutive steps is less than eps.

}

\value{
A list with three elements:
\item{p}{numeric vector. Probability of each node at the steady state or after tmax iterations}
\item{ind.positives}{indices of the "core" positive examples of the graph (it is equal to the same input parameter)}
\item{n.iter}{number of performed steps/iterations}

}

\references{
L. Lovasz, Random Walks on Graphs: a Survey, Combinatorics, Paul Erdos is Eighty, vol. 2, pp. 146, 1993.

}

\seealso{
\code{\link{RWR}}
}

\examples{
# Application of the random walk to the prediction of the DrugBank category Penicillins
# using the Tanimoto chemical structure similarity network 
# between 1253 DrugBank drugs
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
# 2-step random walk
res <- RW(DD.chem.data, ind.pos, tmax = 2);
\dontrun{
# 5 steps random walk
res <- RW(DD.chem.data, ind.pos, tmax = 5);
}
}
