\name{do.RW}
\alias{do.RW}

\title{
Random walk cross-validation experiments with multiple classes
}

\description{
High level function to perform random walk cross-validation experiments with multiple classes.
}

\usage{
do.RW(tmax = 1000, eps = 1e-10, k = 5, stratified=TRUE, filter = TRUE, seed = 1, 
      data.dir, labels.dir, output.dir, data, labels)
}

\arguments{
\item{tmax}{maximum number of iterations (def: 1000)}

\item{eps}{maximum allowed difference between the computed probabilities at the steady state (def. 1e-10)}

\item{k}{number of folds for the cross validation (def. 5)}

\item{stratified}{boolean. If TRUE (def.) stratified CV is performed otherwise vanilla CV is done}

\item{filter}{if TRUE (def) the adjacency matrix is sparsified otherwise not}

\item{seed}{seed of the random generator for the generation of the folds (def: 1)}

\item{data.dir}{relative path to directory where the adjiacency matrix is stored }

\item{labels.dir}{relative path to directory where the label matrix is stored }

\item{output.dir}{relative path to directory where the results are stored  }

\item{data}{name of the data set to loaded (without rda extension). It must be  an .rda file containing the adjacency matrix of the graph.
It assumes that it is in the data.dir directory}

\item{labels}{name of the target labels (without rda extension). It must be  an .rda file containing the label matrix of the examples.
It assumes that it is in the labels.dir directory}
}

\details{
High level function to perform cross-validation experiments with multiple classes using RW.

It performs a k fold CV on a given data set, and output scores, AUC and Precision at a given recall results for multiple classes.

Graph data are read from a matrix representing the adjiacency matrix of the graph stored as a .rda file. The labels are read from a matrix having examples as rows and classes as columns stored as a .rda file. If \eqn{M} is the label matrix, then \eqn{M[i,j]=1}, if example \eqn{i} is annotated with class \eqn{j}, otherwise \eqn{M[i,j] = 0}.

Results are included in matrices representing Scores, AUC and precision at a given recall results stored as .rda files.
}

\value{
3 rda files stored in the Results directory:
\item{Scores results}{A matrix with examples on rows and classes on columns representing the computed scores for each example and for each considered class}
\item{AUC results}{AUC results files computed through \code{AUC.single.over.classes} from the package PerfMeas}
\item{Precision at given recall results}{computed through \code{precision.at.multiple.recall.level.over.classes} from the package PerfMeas.}
The name of the Score file starts with Score, of the AUC file with AUC, and of the Precision at given recall file with PXR.
Other learning parameters are appended to the name of the file. All the results .rda files are stored in the Results directory (that must exist in advance).
}


\seealso{
\code{\link{RW}},  \code{\link{multiple.RW.cv}}, \code{\link{do.RWR}}
}

\examples{
\donttest{
# Yeast prediction of 177 FunCat classes by 5-fold cross validation 
# using 3 steps of Random walk and STRING data. 
# data obtained from the bionetdata package from CRAN
# See the AUC and Precision/recall results in the Results directory
library(bionetdata);
dd=tempdir();
rr=tempdir();
data(Yeast.STRING.data);
data(Yeast.STRING.FunCat);
save(Yeast.STRING.data, file=paste(dd,"/net.rda", sep=""));
save(Yeast.STRING.FunCat, file=paste(dd,"/labels.rda", sep=""));
do.RW(tmax = 3, filter = FALSE, seed = 1, data.dir=dd, labels.dir=dd, 
output.dir=rr, data="/net", labels="/labels");
}
}

