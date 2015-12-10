\name{GBAsum}
\alias{GBAsum}

\title{
Guilt By Association (GBA) using the sum rule
}

\description{
GBAsum implements a Guilt By Association (GBA) method based on the sum of incident edge weights
}

\usage{
GBAsum(W, ind.positives)
}

\arguments{
\item{W}{numeric matrix representing the adjacency matrix of the graph}

\item{ind.positives}{indices of the "core" positive examples of the graph. They represent the indices of W corresponding to the positive examples.}
}

\details{
Function that implements a Guilt By Association (GBA) method for  label ranking based on
the sum of edge weights connecting a node to its positive neighbours.
}

\value{
a list with one element:
\item{p}{score associated to each node}
}

\references{
Oliver, S., Guilt-by-association goes global, Nature, 403, pp. 601-603, 2000.
}


\seealso{
\code{\link{GBAmax}}
}

\examples{
# Application of GBAsum to the prediction of the DrugBank category Penicillins
# using the Tanimoto chemical structure similarity network 
# between 1253 DrugBank drugs
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
GBAsum(DD.chem.data, ind.pos);
# Application of GBAsum to the prediction of the DrugBank category "Anti_HIV_Agents"
labels <- DrugBank.Cat[,"Anti_HIV_Agents"];
ind.pos <- which(labels==1);
GBAsum(DD.chem.data, ind.pos);
}
