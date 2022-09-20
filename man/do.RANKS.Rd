\name{do.RANKS}
\alias{do.RANKS}

\title{
RANKS cross-validation experiments with multiple classes
}

\description{
High level function to perform RANKS cross-validation experiments with multiple classes.
}

\usage{
do.RANKS(score = eav.score, kernel = rw.kernel, a = 2, p = 1, sparsify = TRUE, kk = 5, 
rep = 1, stratified=TRUE, seed = 0, data.dir, labels.dir, 
output.dir, data, labels, ...)
}

\arguments{
\item{score}{function. It must be a kernel-based score method:

- eav.score (default)

- NN.score

- KNN.score

- WSLD.score
}

\item{kernel}{kernel metod or function (def. rw.kernel)}

\item{a}{kernel parameter (def. 2)}

\item{p}{number of steps of the RW kernel (def. 1)}

\item{sparsify}{boolean. If TRUE (def) the input matrix is sparsified using Sparsify.matrix from the package NetpreProc}

\item{kk}{number of folds of the cross validation (def: 5)}

\item{rep}{number of repetitions of the cross validation (def: 1)}

\item{stratified}{boolean. If TRUE (def.) stratified CV is performed otherwise vanilla CV is done}

\item{seed}{initialization seed for the random generator to create folds (def:0)}

\item{data.dir}{relative path to directory where the adjiacency matrix is stored}

\item{labels.dir}{relative path to directory where the label matrix is stored}

\item{output.dir}{relative path to directory where the results are stored}

\item{data}{name of the data set to loaded (without rda extension). It must be  an .rda file containing the adjiacency matrix of the graph.
It assumes that it is in the data.dir directory}

\item{labels}{name of the target labels (without rda extension). It must be  an .rda file containing the label matrix of the examples.
It assumes that it is in the labels.dir directory. Note that data and labels must have the same number of rows and in the same order}

\item{...}{optional arguments to be passed to the function \code{multiple.ker.score.cv} that performs the CV}
}

\details{
High level function to perform cross-validation experiments with multiple classes using RANKS.

It performs a k fold CV repeated multiple times on a given data set, and scores, AUC and Precision at a given recall results for multiple classes are generated.

Graph data are read from a matrix representing the adjiacency matrix of the graph stored as a .rda file. The labels are read from a matrix having examples as rows and classes as columns stored as a .rda file. If \eqn{M} is the label matrix, then \eqn{M[i,j]=1}, if example \eqn{i} is annotated with class \eqn{j}, otherwise \eqn{M[i,j] = 0}.

Results are included in matrices representing Scores, AUC and precision at a given recall results stored as .rda files.
}

\value{
3 rda files stored in the output.dir directory:
\item{Scores results}{A matrix with examples on rows and classes on columns representing the computed scores for each example and for each considered class}
\item{AUC results}{AUC results files computed through \code{AUC.single.over.classes} from the package PerfMeas}
\item{Precision at given recall results}{computed through \code{precision.at.multiple.recall.level.over.classes} from the package PerfMeas.}
The name of the Score file starts with Score, of the AUC file with AUC, and of the Precision at given recall file with PXR.
Other learning parameters are appended to the name of the file.
}

\seealso{
\code{\link{multiple.ker.score.cv}}, \code{\link{do.loo.RANKS}}
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
do.RANKS(data.dir=dd, labels.dir=dd, output.dir=rr, data="/net", labels="/labels");
}
}


