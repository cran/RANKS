\name{RW.cv}
\alias{RW.cv}

\title{
Random walk, GBA and labelprop cross-validation for a single class
}

\description{
Function to execute cross-validation with random walk based, labelprop and GBA methods
}

\usage{
RW.cv(W, ind.pos, k = 5, stratified=TRUE, init.seed = 0, fun = RW, ...)
}

\arguments{
\item{W}{a numeric matrix representing the adjacency matrix of the graph.
Note that if the optional argument norm=TRUE (def.), the W matrix is normalized, otherwise it
is assumed that W is just normalized}

\item{ind.pos}{indices of the "core" positive examples of the graph. They represent the indices of W corresponding to the positive examples}

\item{k}{number of folds (def: 5)}

\item{stratified}{boolean. If TRUE (def.) stratified CV is performed otherwise vanilla CV is done}

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
It performs a single cross-validation for a single class. It can be used with of the following methods: RW, RWR, label.prop, GBAsum, GBAmax.
}

\value{
a vector with the the probabilities for each example at the steady state
}


\seealso{
\code{\link{RW}}, \code{\link{RWR}}, \code{\link{label.prop}}, \code{\link{GBAsum}}, \code{\link{GBAmax}},  \code{\link{multiple.RW.cv}} 
}

\examples{
# Nodel label ranking of the DrugBank category Penicillins
# on the Tanimoto chemical structure similarity network (1253 drugs)
# using 5 fold cross-validation and GBAsum
library(bionetdata);
data(DD.chem.data);
data(DrugBank.Cat);
labels <- DrugBank.Cat[,"Penicillins"];
ind.pos <- which(labels==1);
res <- RW.cv(DD.chem.data, ind.pos, k = 5, init.seed = 0, fun = GBAsum);

\donttest{
# the same but using label.prop
res <- RW.cv(DD.chem.data, ind.pos, k = 5, init.seed = 0, fun = label.prop, tmax=2);

# the same but using "vanilla" 2-step random walk
res <- RW.cv(DD.chem.data, ind.pos, k = 5, init.seed = 0, fun = RW, tmax=2);
}
}
