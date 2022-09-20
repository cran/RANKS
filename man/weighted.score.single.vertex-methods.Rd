\name{weighted.score.single.vertex-methods}
\docType{methods}
\alias{Methods for scoring a single vertex - weighted version}
\alias{single.NN.w.score-methods}
\alias{single.NN.w.score,graph-method}
\alias{single.NN.w.score,matrix-method}
\alias{single.KNN.w.score-methods}
\alias{single.KNN.w.score,graph-method}
\alias{single.KNN.w.score,matrix-method}
\alias{single.eav.w.score-methods}
\alias{single.eav.w.score,graph-method}
\alias{single.eav.w.score,matrix-method}
\alias{single.NN.w.score}
\alias{single.KNN.w.score}
\alias{single.eav.w.score}



\title{Single vertex score functions - weighted version}

\description{
  Methods to compute weighted score functions applied to a single vertex of the graph
}

\usage{
\S4method{single.NN.w.score}{graph}(RW, x, x.pos, w)
\S4method{single.NN.w.score}{matrix}(RW, x, x.pos, w)
\S4method{single.KNN.w.score}{graph}(RW, x, x.pos, w, k = 3)
\S4method{single.KNN.w.score}{matrix}(RW, x, x.pos, w, k = 3)
\S4method{single.eav.w.score}{graph}(RW, x, x.pos, w, auto = FALSE)
\S4method{single.eav.w.score}{matrix}(RW, x, x.pos, w, auto = FALSE)
}



\arguments{
\item{RW}{matrix. It must be a kernel matrix or a symmetric matrix expressing the similarity between nodes}

\item{x}{integer. Index corresponding to the element of the RW matrix for which the score must be computed}

\item{x.pos}{vector of integer. Indices of the positive elements of the RW matrix}

\item{w}{vector of numeric. Its elements represent the initial likelihood that the nodes of the graph belong to the class under study. 
The elements of w correspond to the columns of RW and the length of w and the number of columns of RW must be equal.}

\item{k}{integer. Number of the k nearest neighbours to be considered}

\item{auto}{boolean. If TRUE the components \eqn{K(x,x) + K(x_i,x_i)} are computed, otherwise are discarded (default)}

}



\value{
\code{single.NN.w.score}: the weighted NN score of the vertex

\code{single.KNN.w.score}:  the weighted KNN score of the vertex

\code{single.eav.w.score}: the  weighted Empirical Average score of the vertex

}

\details{
\code{single.NN.w.score} computes the weighted NN score for a single vertex:

\deqn{score(x) = - \min_{x_i \in V_C} -2 K(x,x_i)) * w(x_i)}

where \eqn{V_C} is the set of positive vertices, and \eqn{w(x_i)} is the weight associated to the node \eqn{x_i}

\code{single.KNN.w.score} compute the weighted KNN score for a single vertex:

\deqn{score(x) = \sum_{k \; nearest \; x_i \in V_C}  2 K(x,x_i) * w(x_i)}

\code{single.eav.score} computes the weighted Empirical Average score for a single vertex:

\deqn{score(x) = - K(x,x) * w(x) + \frac{2}{(\sum_{x_i \in x.pos} w(x_i))} * \sum_{x_i \in x.pos} K(x,x_i) * w(x_i)}


}

\section{Methods}{
\describe{



\item{\code{signature(RW = "graph")}}{
 \code{single.NN.w.score} computes the weighted NN score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
  \code{single.KNN.w.score} computes the  weighted KNN score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{single.eav.w.score} computes the  weighted Empirical Average score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
}

\item{\code{signature(RW = "matrix")}}{
  \code{single.NN.w.score} computes the  weighted NN score for a single vertex using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{single.KNN.w.score} computes the  weighted KNN score for a single vertex using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{single.eav.score} computes the weighted  Empirical Average score using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
}
}
}


\seealso{
\code{\link{Methods for scoring a single vertex}}
\code{\link{Methods for scoring multiple vertices - weighted version}}
}




\examples{
# Computation of scores using STRING data with respect to 
# the FunCat category 11.02.01 rRNA synthesis 
library(bionetdata);
data(Yeast.STRING.data);
data(Yeast.STRING.FunCat);
labels <- Yeast.STRING.FunCat[,"11.02.01"];
n <- length(labels);
ind.pos <- which(labels==1);
# NN-score  computed directly on the STRING matrix on the first yeast gene YJR121W
s <- single.NN.w.score(Yeast.STRING.data, 1, ind.pos, w=labels);
# NN-score weighted computed directly on the STRING matrix on the first yeast gene YJR121W,
# using this time random weights for the value of positive nodes
w <- runif(n);
s <- single.NN.w.score(Yeast.STRING.data, 1, ind.pos, w=w);
\donttest{
# NN-score weighted computed on the 1 step and 2-step random walk kernel matrix
K <- rw.kernel(Yeast.STRING.data);
sK <- single.NN.w.score(K, 1, ind.pos, w);
K2 <- p.step.rw.kernel(K, p=2);
sK2 <- single.NN.w.score(K2, 1, ind.pos, w);
}
}

