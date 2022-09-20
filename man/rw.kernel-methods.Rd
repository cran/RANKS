\name{rw.kernel-methods}
\docType{methods}
\alias{rw.kernel-methods}
\alias{rw.kernel,graph-method}
\alias{rw.kernel,matrix-method}
\alias{p.step.rw.kernel-methods}
\alias{p.step.rw.kernel,graph-method}
\alias{p.step.rw.kernel,matrix-method}
\alias{rw.kernel}
\alias{p.step.rw.kernel}

\title{ Random walk kernel}
\description{
  Methods to compute the random walk kernel (Smola and Kondor, 2003)
}

\usage{
\S4method{rw.kernel}{matrix}(W, a = 2)
\S4method{rw.kernel}{graph}(W, a = 2)
\S4method{p.step.rw.kernel}{graph}(RW, p = 2)
\S4method{p.step.rw.kernel}{matrix}(RW, p = 2)
}



\arguments{

\item{W}{a square symmetric matrix with positive values or an object of the class graphAM  or graphNEL of the package graph}

\item{RW}{matrix. It must be a random walk kernel matrix}

\item{a}{numeric. It is correlated to the probability of remaining at the same vertex. Larger a, larger the probability (def. 2)}

\item{p}{ integer. Number of steps (def: p=2)}
}


\value{
\code{rw.kernel}: A numeric square matrix representing a one-step random walk kernel matrix 

\code{p.step.rw.kernel}:  A numeric square matrix representing a p-step random walk kernel matrix

}

\details{
\code{rw.kernel} methods computes the one step random walk kernel RW, i.e.:
\deqn{
RW = (a-1)I + D^{-\frac{1}{2}} * W * D^{-\frac{1}{2}}
}

where \eqn{I} is the identity matrix, \eqn{W} is the weighted adjacency matrix of an undirected graph,
and  \eqn{D} is a diagonal matrix with \eqn{D_{ii} = \sum_j W_{ij}}

\code{p.step.rw.kernel} methods compute the p-step random walk kernel pRW, i.e.:
\deqn{
pRW = RW^p
}
}

\section{Methods}{

\describe{

\item{\code{signature(W = "graph")}}{
 \code{rw.kernel} computes the random walk kernel starting from a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph) 
}

\item{\code{signature(W = "matrix")}}{
  \code{rw.kernel} computes the random walk kernel starting from a weighted adjacency matrix representing the graph
}

\item{\code{signature(RW = "graph")}}{
 \code{p.step.rw.kernel} computes the a p-step random walk kernel starting from a graph of class graph 
 (hence including objects of class graphAM  and graphNEL from the package graph) 
}

\item{\code{signature(RW = "matrix")}}{
  \code{p.step.rw.kernel} computes the p-step random walk kernel starting from a one-step random walk kernel matrix
}
}
}

\examples{
# Random walk kernel computation using Functional Interaction network data
library(bionetdata);
data(FIN.data);
W <- as.matrix(FIN.data);
K <- rw.kernel(W);
# this a 2-step random walk kernel
\donttest{
K2 <- p.step.rw.kernel(K, p=2);}
}
