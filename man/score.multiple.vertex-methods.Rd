\name{score.multiple.vertex-methods}
\docType{methods}
\alias{Methods for scoring multiple vertices}
\alias{NN.score-methods}
\alias{NN.score,graph-method}
\alias{NN.score,matrix-method}
\alias{KNN.score-methods}
\alias{KNN.score,graph-method}
\alias{KNN.score,matrix-method}
\alias{eav.score-methods}
\alias{eav.score,graph-method}
\alias{eav.score,matrix-method}
\alias{WSLD.score-methods}
\alias{WSLD.score,graph-method}
\alias{WSLD.score,matrix-method}
\alias{NN.score}
\alias{KNN.score}
\alias{eav.score}
\alias{WSLD.score}

\title{Multiple vertex score functions}

\description{
  Methods to compute score functions for multiple vertices of the graph
}

\usage{
\S4method{NN.score}{graph}(RW, x, x.pos, auto = FALSE, norm = TRUE)
\S4method{NN.score}{matrix}(RW, x, x.pos, auto = FALSE, norm = TRUE)
\S4method{KNN.score}{graph}(RW, x, x.pos, k = 3, auto = FALSE, norm = TRUE)
\S4method{KNN.score}{matrix}(RW, x, x.pos, k = 3, auto = FALSE, norm = TRUE)
\S4method{eav.score}{graph}(RW, x, x.pos, auto = FALSE, norm = TRUE)
\S4method{eav.score}{matrix}(RW, x, x.pos, auto = FALSE, norm = TRUE)
\S4method{WSLD.score}{graph}(RW, x, x.pos, d = 2, auto = FALSE, norm = TRUE)
\S4method{WSLD.score}{matrix}(RW, x, x.pos, d = 2, auto = FALSE, norm = TRUE)
}



\arguments{
\item{RW}{matrix. It must be a kernel matrix or a symmetric matrix expressing the similarity between nodes}

\item{x}{vector of integer. Indices corresponding to the elements of the RW matrix for which the score must be computed}

\item{x.pos}{vector of integer. Indices of the positive elements of the RW matrix}

\item{k}{integer. Number of the k nearest neighbours to be considered}

\item{d}{integer. Coefficient of linear decay (def. 2)}

\item{auto}{boolean. If TRUE the components \eqn{K(x,x) + K(x_i,x_i)} are computed, otherwise are discarded (default)}

\item{norm}{boolean. If TRUE (def.) the scores are normalized between 0 and 1.}

}


\value{
\code{NN.score}: a numeric vector with the NN scores of the vertices. The names of the vector correspond to the indices x

\code{KNN.score}:  a numeric vector with the KNN scores of the vertices. The names of the vector correspond to the indices x

\code{eav.score}: a numeric vector with the  Empirical Average score of the vertices. The names of the vector correspond to the indices x

\code{WSLD.score}: a numeric vector with the Weighted Sum with Linear Decay score (WSLD) of the vertices. The names of the vector correspond to the indices x

}

\details{
The methods compute the scores for multiple vertices according to NN, KNN, Empirical Average or WSLD score (see reference for bibliographic details).
Note that the argument x indicates the set of nodes for which the score must be computed. 
The vector x represents the indices of the rows of the matrix RW corresponding to the vertices for which the scores must be computed. 
If x = 1:nrow(RW) the scores for all the vertices of the graph are computed. 

}

\section{Methods}{
\describe{

\item{\code{signature(RW = "graph")}}{
 \code{NN.score} computes the  NN score for multiple vertices using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
  \code{KNN.score} computes the  KNN score for multiple vertices using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{eav.score} computes the  Empirical Average score for multiple verticesusing a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{WSLD.score} computes the  Weighted Sum with Linear Decay score for multiple vertices using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
}

\item{\code{signature(RW = "matrix")}}{
  \code{NN.score} computes the  NN score for multiple vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{KNN.score} computes the  KNN score for multiple vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{eav.score} computes the  Empirical Average score multiple for vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{WSLD.score} computes the  Weighted Sum with Linear Decay score for multiple vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
}
}
}

\seealso{
\code{\link{Methods for scoring a single vertex}}
}

\references{
Re M, Mesiti M, Valentini G: A fast ranking algorithm for predicting gene functions in biomolecular networks.
IEEE ACM Trans Comput Biol Bioinform 2012, 9(6):1812-1818.

Insuk Lee, Bindu Ambaru, Pranjali Thakkar, Edward M. Marcotte, and Seung Y. Rhee. Nature Biotechnology 28, 149-156, 2010
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
# NN-scores computed directly on the STRING matrix 
s <- NN.score(Yeast.STRING.data, 1:n, ind.pos);
\dontrun{
# NN-scores computed on the 1 step and 2-step random walk kernel matrix
K <- rw.kernel(Yeast.STRING.data);
sK <- NN.score(K, 1:n, ind.pos);
K2 <- p.step.rw.kernel(K, p=2);
sK2 <- NN.score(K2, 1:n, ind.pos);
# WSLD-scores computed directly on the STRING matrix 
s <- WSLD.score(Yeast.STRING.data, 1:n, ind.pos);
# WSLD-scores computed on the 1 step and 2-step random walk kernel matrix
sK <- WSLD.score(K, 1:n, ind.pos);
sK2 <- WSLD.score(K2, 1:n, ind.pos);
}
}

