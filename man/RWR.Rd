\name{RWR}
\alias{RWR}

\title{
Random walk with Restart on a graph
}

\description{
Function that performs a random Walk with restart (RWR) on a given graph
}

\usage{
RWR(W, ind.positives, gamma = 0.6, tmax = 1000, eps = 1e-10, norm = TRUE)
}

\arguments{
\item{W}{a numeric matrix representing the adjacency matrix of the graph}

\item{ind.positives}{indices of the "core" positive examples of the graph.
They represent the indices of W corresponding to the positive examples}

\item{gamma}{restart parameter (def: 0.6)}

\item{tmax}{maximum number of iterations (steps) (def: 1000)}

\item{eps}{maximum allowed difference between the computed probabilities at the steady state (def. 1e-10)}

\item{norm}{if TRUE (def) the adjacency matrix \eqn{W} of the graph is normalized to \eqn{M = D^{-1} * W}, otherwise it is assumed that the matrix \eqn{W} is just normalized}
}

\details{
RWR performs a random Walk with restart on a given graph by performing 1 or more steps on the graph, depending on the value of the tmax parameter.
The restart parameter expresses the probability of "restarting" from a "core" node at each step of the random walk algorithm.
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
\code{\link{RW}}
}

\examples{
# Application of the random walk with restart to the prediction of the 
# DrugBank category Penicillins
# using the Tanimoto chemical structure similarity network 
# between 1253 DrugBank drugs
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
# 2-step RWR
res <- RWR(DD.chem.data, ind.pos, tmax = 2);
\donttest{
# till to convergence
res <- RWR(DD.chem.data, ind.pos, tmax = 5000, eps=1e-6);
# 5 steps and higher gamma
res <- RWR(DD.chem.data, ind.pos, tmax = 5, gamma=0.8);
}
}
