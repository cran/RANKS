\name{score.single.vertex-methods}
\docType{methods}
\alias{Methods for scoring a single vertex}
\alias{single.NN.score-methods}
\alias{single.NN.score,graph-method}
\alias{single.NN.score,matrix-method}
\alias{single.KNN.score-methods}
\alias{single.KNN.score,graph-method}
\alias{single.KNN.score,matrix-method}
\alias{single.eav.score-methods}
\alias{single.eav.score,graph-method}
\alias{single.eav.score,matrix-method}
\alias{single.WSLD.score-methods}
\alias{single.WSLD.score,graph-method}
\alias{single.WSLD.score,matrix-method}
\alias{single.NN.score}
\alias{single.KNN.score}
\alias{single.eav.score}
\alias{single.WSLD.score}

\title{Single vertex score functions}

\description{
  Methods to compute score functions applied to a single vertex of the graph
}

\usage{
\S4method{single.NN.score}{graph}(RW, x, x.pos, auto = FALSE)
\S4method{single.NN.score}{matrix}(RW, x, x.pos, auto = FALSE)
\S4method{single.KNN.score}{graph}(RW, x, x.pos, k = 3, auto = FALSE)
\S4method{single.KNN.score}{matrix}(RW, x, x.pos, k = 3, auto = FALSE)
\S4method{single.eav.score}{graph}(RW, x, x.pos, auto = FALSE)
\S4method{single.eav.score}{matrix}(RW, x, x.pos, auto = FALSE)
\S4method{single.WSLD.score}{graph}(RW, x, x.pos, d = 2, auto = FALSE)
\S4method{single.WSLD.score}{matrix}(RW, x, x.pos, d = 2, auto = FALSE)
}



\arguments{
\item{RW}{matrix. It must be a kernel matrix or a symmetric matrix expressing the similarity between nodes}

\item{x}{integer. Index corresponding to the element of the RW matrix for which the score must be computed}

\item{x.pos}{vector of integer. Indices of the positive elements of the RW matrix}

\item{k}{integer. Number of the k nearest neighbours to be considered}

\item{d}{integer. Coefficient of linear decay (def. 2)}

\item{auto}{boolean. If TRUE the components \eqn{K(x,x) + K(x_i,x_i)} are computed, otherwise are discarded (default)}

}


\value{
\code{single.NN.score}: the NN score of the vertex

\code{single.KNN.score}:  the KNN score of the vertex

\code{single.eav.score}: the  Empirical Average score of the vertex

\code{single.WSLD.score}: the  Weighted Sum with Linear Decay score (WSLD) of the vertex

}

\details{
\code{single.NN.score} computes the  NN score for a single vertex:

\deqn{score(x) = - \min_{x_i \in V_C} ( K(x,x) + K(x_i,x_i) -2 K(x,x_i))}

where \eqn{V_C} is the set of positive vertices.

\code{single.KNN.score} compute KNN score for a single vertex:

\deqn{score(x) = - \sum_{k \; nearest \; x_i \in V_C} ( K(x,x) + K(x_i,x_i) - 2 K(x,x_i))}

\code{single.eav.score} computes the Empirical Average score for a single vertex:

\deqn{score(x) = - K(x,x) + \frac{2}{|V_C|} * \sum_{x_i \in V_C} K(x,x_i)}

\code{single.WSLD.score} computes the  WSLD score for a single vertex:

Let \eqn{K(x, x_{jk})} be the kth rank order index w.r.t. \eqn{x_j \in V_C}, and \eqn{m=|V_C|}, then:
\deqn{score(x) = \max_{x_i \in V_C} K(x,x_i) + \sum_{k=2}^m [(1/(d * (k-1))) * K(x, x_{jk})]}


}

\section{Methods}{
\describe{



\item{\code{signature(RW = "graph")}}{
 \code{single.NN.score} computes the  NN score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
  \code{single.KNN.score} computes the  KNN score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{single.eav.score} computes the  Empirical Average score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{single.WSLD.score} computes the  Weighted Sum with Linear Decay score for a single vertex using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
}

\item{\code{signature(RW = "matrix")}}{
  \code{single.NN.score} computes the  NN score for a single vertex using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{single.KNN.score} computes the  KNN score for a single vertex using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{single.eav.score} computes the  Empirical Average score using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{single.WSLD.score} computes the  Weighted Sum with Linear Decay score for a single vertex using a kernel matrix or a symmetric matrix expressing the similarity between nodes
}
}
}


\seealso{
\code{\link{Methods for scoring multiple vertices}}
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
# NN-score computed directly on the STRING matrix on the first yeast gene YJR121W
s <- single.NN.score(Yeast.STRING.data, 1, ind.pos);
\dontrun{
# NN-score computed on the 1 step and 2-step random walk kernel matrix
K <- rw.kernel(Yeast.STRING.data);
sK <- single.NN.score(K, 1, ind.pos);
K2 <- p.step.rw.kernel(K, p=2);
sK2 <- single.NN.score(K2, 1, ind.pos);

# WSLD-score computed directly on the STRING matrix on the first  yeast gene YJR121W
s <- single.WSLD.score(Yeast.STRING.data, 1, ind.pos);
# WSLD-scores computed on the 1 step and 2-step random walk kernel matrix
sK <- single.WSLD.score(K, 1, ind.pos);
sK2 <- single.WSLD.score(K2, 1, ind.pos);
}
}

