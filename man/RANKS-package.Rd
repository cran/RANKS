\name{RANKS-package}
\alias{RANKS-package}
\alias{RANKS}
\docType{package}
\title{
RANKS: Ranking of Nodes with Kernelized Score Functions
}

\description{
Implementation of Kernelized score functions and other semi-supervised learning algorithms for node label ranking 
in biomolecular networks. 
}

\details{
RANKS can be easily applied to a large set of different relevant problems in computational biology, 
ranging from automatic protein function prediction, to gene disease prioritization and drug repositioning, and more in general 
to any bioinformatics problem that can be formalized as a node label ranking problem in a graph. 
The modular nature of the implementation allows to experiment with different score functions and kernels and to easily 
compare the results with baseline network-based methods such as label propagation and random walk algorithms, as well 
as to enlarge the algorithmic scheme by adding novel user-defined score functions and kernels.
}

\author{
\emph{Giorgio Valentini} 

AnacletoLab

DI, Dipartimento di Informatica

Universita' degli Studi di Milano

\email{valentini@di.unimi.it}

Maintainer: 
\emph{Giorgio Valentini} 
}

\references{
Giorgio Valentini, Giuliano Armano, Marco Frasca, Jianyi Lin, Marco Mesiti, and Matteo Re 
RANKS: a flexible tool for node label ranking and classification in biological networks
Bioinformatics first published online June 2, 2016 doi:10.1093/bioinformatics/btw235 

Re M, Mesiti M, Valentini G: A fast ranking algorithm for predicting gene functions in biomolecular networks.
IEEE ACM Trans Comput Biol Bioinform 2012, 9(6):1812-1818.

Re M, Valentini G: Cancer module genes ranking using kernelized score functions.
BMC Bioinformatics 2012, 13(S14):S3.

Re M, Valentini G: Network-based drug ranking and repositioning with respect to DrugBank therapeutic categories.
IEEE/ACM Trans Comput Biol Bioinform 2013, 10(6):1359-1371. 

G. Valentini, A. Paccanaro, H. Caniza, A. Romero, M. Re: An extensive analysis of disease-gene associations using network integration
and fast kernel-based gene prioritization methods, Artif. Intell. in Med. 61 (2) (2014) 63-78
}

\keyword{ package }
