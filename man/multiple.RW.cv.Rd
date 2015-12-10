\name{multiple.RW.cv}
\alias{multiple.RW.cv}

\title{
Random walk, GBA and labelprop multiple cross-validation for a single class
}

\description{
Function to execute multiple cross-validation with random walk based, labelprop and GBA methods
}

\usage{
multiple.RW.cv(W, ind.pos, k = 5, p = 100, init.seed = 0, fun = RW, ...)
}

\arguments{
\item{W}{a numeric matrix representing the adjacency matrix of the graph.
Note that if the optional argument norm=TRUE (def.), the W matrix is normalized, otherwise it
is assumed that W is just normalized}

\item{ind.pos}{indices of the "core" positive examples of the graph. They represent the indices of W corresponding to the positive examples}

\item{k}{number of folds (def: 5)}

\item{p}{number of repeated cross-validations}

\item{init.seed}{initial seed for the random generator. If 0 (default) no initialization is performed}

\item{fun}{function. It must be one of the following functions:

- RW (default)

- RWR

- label.prop

- GBAsum

- GBAmax
}

\item{...}{optional arguments for the function fun:

- gamma : restart parameter (def: 0.6) (meaningful only for RWR)

- tmax : maximum number of iterations (def: 1000)

- eps : maximum allowed difference between the computed probabilities at the steady state (def. 1e-10)
}
}

\details{
Function to execute multiple cross-validation with random walk based, labelprop and GBA methods for a single class.
It computes the scores by averaging across multiple cross validations.
It can be used with of the following methods: RW, RWR, label.prop, GBAsum, GBAmax.
}

\value{
a vector with the the probabilities for each example at the steady state averaged across multiple cross-validations
}



\seealso{
\code{\link{RW}}, \code{\link{RWR}}, \code{\link{label.prop}}, \code{\link{GBAsum}}, \code{\link{GBAmax}},  \code{\link{RW.cv}} 
}

\examples{
# Nodel label ranking of the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation repeated 2 times
# and "vanilla" 2-step random walk
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);

res <- multiple.RW.cv(DD.chem.data, ind.pos, k = 5, p = 2, init.seed = 0, fun = GBAmax)

\dontrun{
# the same but using the label.prop
res <- multiple.RW.cv(DD.chem.data, ind.pos, k = 5, p = 2, init.seed = 0, fun = label.prop, tmax=2)

# the same but using "vanilla" 2-step random walk
res <- multiple.RW.cv(DD.chem.data, ind.pos, k = 5, p = 2, init.seed = 0, fun = RW, tmax=2)
}

}
