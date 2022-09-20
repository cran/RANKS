\name{do.GBA}
\alias{do.GBA}

\title{
GBA cross-validation experiments with multiple classes
}

\description{
High level function to perform experiments with GBA.
It perform a k fold CV repeated 1 time on a given data set
}

\usage{
do.GBA(fun = GBAsum, k = 5, stratified=TRUE, filter = TRUE,  seed = 1, 
       data.dir, labels.dir, output.dir, data, labels)
}

\arguments{
\item{fun}{function performing GBA. it can be one of the following:

- GBAsum: it sums the edge weights connecting a node to its positive neighbours

- GBAmax: it computes the maximum between the edge weights connecting a node to its positive neighbours}

\item{k}{number of folds for the cross validation (def. 5)}

\item{stratified}{boolean. If TRUE (def.) stratified CV is performed otherwise vanilla CV is done} 

\item{filter}{if TRUE (def) the adjacency  matrix is sparsified otherwise not}

\item{seed}{seed of the random generator for the generation of the folds (def: 1)}

\item{data.dir}{relative path to directory where the adjiacency matrix is stored }

\item{labels.dir}{relative path to directory where the label matrix is stored }

\item{output.dir}{relative path to directory where the results are stored }

\item{data}{name of the data set to loaded (without rda extension). It must be  an .rda file containing the adjiacency matrix of the graph.
It assumes that it is in the "data.dir" directory}

\item{labels}{name of the target labels (without rda extension). It must be  an .rda file containing the label matrix of the examples.
Rows correspond to examples and columns to classes
It assumes that it is in the "labels.dir" directory}
}

\details{
High level function to perform cross-validation experiments with multiple classes using GBA.

It performs a k fold CV on a given data set, and output scores, AUC and Precision at a given recall results for multiple classes.

Graph data are read from a matrix representing the adjiacency matrix of the graph stored as a .rda file. The labels are read from a matrix having examples as rows and classes as columns stored as a .rda file. If M is the label matrix, then M[i,j]=1, if example i is annotated with class j, otherwise M[i,j] = 0.

Results are included in matrices representing Scores, AUC and precision at a given recall results stored as .rda files.
}

\value{
3 rda files stored in the "Results" directory:
\item{Scores results}{A matrix with examples on rows and classes on columns representing the computed scores for each example and for each considered class}
\item{AUC results}{AUC results files computed through \code{AUC.single.over.classes} from the package PerfMeas}
\item{Precision at given recall results}{computed through \code{precision.at.multiple.recall.level.over.classes} from the package PerfMeas.}
The name of the Score file starts with Score, of the AUC file with AUC, and of the Precision at given recall file with PXR.
Other learning parameters are appended to the name of the file. All the results .rda files are stored in the Results directory (that must exist in advance).
}


\seealso{
\code{\link{GBAmax}}, \code{\link{GBAsum}}
}

\examples{
\donttest{
# Yeast prediction of 177 FunCat classes by 5-fold cross validation using STRING data
# data obtained from the bionetdata package from CRAN
# See the AUC and Precision/recall results in the Results directory
library(bionetdata);
dd=tempdir();
rr=tempdir();
data(Yeast.STRING.data);
data(Yeast.STRING.FunCat);
save(Yeast.STRING.data, file=paste(dd,"/net.rda", sep=""));
save(Yeast.STRING.FunCat, file=paste(dd,"/labels.rda", sep=""));
do.GBA(data.dir=dd, labels.dir=dd, output.dir=rr, data="/net", labels="/labels");
}
}


