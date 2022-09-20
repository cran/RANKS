\name{weighted.score.multiple.vertex-methods}
\docType{methods}
\alias{Methods for scoring multiple vertices - weighted version}
\alias{NN.w.score-methods}
\alias{NN.w.score,graph-method}
\alias{NN.w.score,matrix-method}
\alias{KNN.w.score-methods}
\alias{KNN.w.score,graph-method}
\alias{KNN.w.score,matrix-method}
\alias{eav.w.score-methods}
\alias{eav.w.score,graph-method}
\alias{eav.w.score,matrix-method}
\alias{NN.w.score}
\alias{KNN.w.score}
\alias{eav.w.score}

\title{Multiple vertex score functions - weighted version}

\description{
  Methods to compute weighted score functions for multiple vertices of the graph
}

\usage{
\S4method{NN.w.score}{graph}(RW, x, x.pos, w, norm = TRUE)
\S4method{NN.w.score}{matrix}(RW, x, x.pos, w, norm = TRUE)
\S4method{KNN.w.score}{graph}(RW, x, x.pos, w, k = 3, norm = TRUE)
\S4method{KNN.w.score}{matrix}(RW, x, x.pos, w, k = 3, norm = TRUE)
\S4method{eav.w.score}{graph}(RW, x, x.pos, w, auto = FALSE, norm = TRUE)
\S4method{eav.w.score}{matrix}(RW, x, x.pos, w, auto = FALSE, norm = TRUE)
}



\arguments{
\item{RW}{matrix. It must be a kernel matrix or a symmetric matrix expressing the similarity between nodes}

\item{x}{vector of integer. Indices corresponding to the elements of the RW matrix for which the score must be computed}

\item{x.pos}{vector of integer. Indices of the positive elements of the RW matrix}

\item{k}{integer. Number of the k nearest neighbours to be considered}

\item{w}{vector of numeric. Its elements represent the initial likelihood that the nodes of the graph belong to the class under study. 
The elements of w correspond to the columns of RW and the length of w and the number of columns of RW must be equal.}

\item{auto}{boolean. If TRUE the components \eqn{K(x,x) + K(x_i,x_i)} are computed, otherwise are discarded (default)}

\item{norm}{boolean. If TRUE (def.) the scores are normalized between 0 and 1.}

}


\value{
\code{NN.w.score}: a numeric vector with the weighted NN scores of the vertices. The names of the vector correspond to the indices x

\code{KNN.score}:  a numeric vector with the weighted KNN scores of the vertices. The names of the vector correspond to the indices x

\code{eav.score}: a numeric vector with the  weighted Empirical Average score of the vertices. The names of the vector correspond to the indices x

}

\details{
The methods compute the weighted scores for multiple vertices according to the weighted version of NN, KNN, and Empirical Average  score.
Note that the argument x indicates the set of nodes for which the score must be computed. 
The vector x represents the indices of the rows of the matrix RW corresponding to the vertices for which the scores must be computed. 
If x = 1:nrow(RW) the scores for all the vertices of the graph are computed. 

}

\section{Methods}{
\describe{

\item{\code{signature(RW = "graph")}}{
 \code{NN.w.score} computes the weighted NN score for multiple vertices using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
  \code{KNN.w.score} computes the  weighted KNN score for multiple vertices using a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
 \code{eav.w.score} computes the  weighted Empirical Average score for multiple verticesusing a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph)
 
}

\item{\code{signature(RW = "matrix")}}{
  \code{NN.w.score} computes the  weighted NN score for multiple vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{KNN.w.score} computes the  weighted KNN score for multiple vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
  \code{eav.w.score} computes the  weighted Empirical Average score multiple for vertices using a kernel matrix or a symmetric matrix expressing the similarity between nodes
  
}
}
}

\seealso{
\code{\link{Methods for scoring a single vertex - weighted version}}
\code{\link{Methods for scoring multiple vertices}}
}

\references{
Giorgio Valentini, Giuliano Armano, Marco Frasca, Jianyi Lin, Marco Mesiti, and Matteo Re 
RANKS: a flexible tool for node label ranking and classification in biological networks
Bioinformatics first published online June 2, 2016 doi:10.1093/bioinformatics/btw235 

Re M, Mesiti M, Valentini G: A fast ranking algorithm for predicting gene functions in biomolecular networks.
IEEE ACM Trans Comput Biol Bioinform 2012, 9(6):1812-1818.

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
s <- NN.w.score(Yeast.STRING.data, 1:n, ind.pos, w=labels);
# Weighted NN-scores computed directly on the STRING matrix 
# using this time random weights for the value of positive nodes
w <- runif(n);
s <- NN.w.score(Yeast.STRING.data, 1:n, ind.pos, w=w);
\donttest{
# Weighted NN-scores computed on the 1 step and 2-step random walk kernel matrix
K <- rw.kernel(Yeast.STRING.data);
sK <- NN.w.score(K, 1:n, ind.pos, w);
K2 <- p.step.rw.kernel(K, p=2);
sK2 <- NN.w.score(K2, 1:n, ind.pos, w);
}
}

