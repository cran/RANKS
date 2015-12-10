\name{do.loo.RANKS}
\alias{do.loo.RANKS}

\title{
RANKS leave-one-out experiments with multiple classes
}
\description{
High level function to perform RANKS leave one out (loo) experiments with mutliple classes.
}

\usage{
do.loo.RANKS(score = eav.score, compute.kernel = TRUE, kernel = rw.kernel, a = 2, 
k = 19, d = 2, p = 1, sparsify = FALSE, norm = FALSE, data, labels, output.name, 
net.dir = "data/", labels.dir = "data/", output.dir = "Results/")
}

\arguments{
\item{score}{function. It must be a kernel-based score method:

- eav.score (default)

- NN.score

- KNN.score

- WSLD.score
}

\item{compute.kernel}{logical. If TRUE (def.) a kernel matrix is computed from data according to the choice of the function kernel, otherwise the data matrix is used as it is.}

\item{kernel}{kernel method or function (def. rw.kernel)}

\item{a}{kernel parameter (def. 2)}

\item{k}{number of neighbours for KNN.score. It is meaningful only for  kNN  (def.19)}

\item{d}{integer. Coefficient of linear decay for the WSLD score. It is meaningful only for  the WSLD score  (def.2)}

\item{p}{number of steps of the RW kernel (def. 1)}

\item{sparsify}{boolean. If TRUE the input matrix is sparsified using Sparsify.matrix from the package NetpreProc (def: FALSE)}

\item{norm}{logical. If TRUE for each class the score is normalized in [0,1], otherwise the raw scores are maintained (default).}

\item{data}{name of the network data set to be loaded (without rda extension). It must be  an .rda file containing the adjiacency matrix of the graph.
By default it assumes that it is in the "data" directory}

\item{labels}{name of the target labels (without rda extension). It must be  an .rda file containing the label matrix of the examples.
By default it assumes that it is in the "data" directory}

\item{output.name}{name of the output file (without rda extension). Other informations including the learning parameters are added in the name of the file}

\item{net.dir}{relative path to the directory where the adjiacency matrix is stored (def: data)}

\item{labels.dir}{relative path to directory where the label matrix is stored (def: data)}

\item{output.dir}{relative path to directory where the results are stored  (def: Results). Note that data and labels must have the same number of rows and in the same order. Moreover if any label column corresponds to any GO root term, this is eliminated to avoid prediction of GO root nodes.}
}

\details{
High level function to perform loo experiments with multiple classes using RANKS.

It performs a loo on a given data set, and scores, AUC and Precision at a given recall results for multiple classes are generated.

Graph data are read from a matrix representing the adjiacency matrix of the graph stored as a .rda file. The labels are read from a matrix having examples as rows and classes as columns stored as a .rda file. If \eqn{M} is the label matrix, then \eqn{M[i,j]=1}, if example \eqn{i} is annotated with class \eqn{j}, otherwise \eqn{M[i,j] = 0}.

Results are included in matrices representing Scores, AUC and precision at a given recall results stored as .rda files.
}

\value{
3 rda files stored in the output.dir directory:
\item{Scores results}{A matrix with examples on rows and classes on columns representing the computed scores for each example and for each considered class}
\item{AUC results}{AUC results files computed through \code{AUC.single.over.classes} from the package PerfMeas}
\item{Precision at given recall results}{computed through \code{precision.at.multiple.recall.level.over.classes} from the package PerfMeas.}
The name of the Score file starts with Score.loo, of the AUC file with AUC.loo, and of the Precision at given recall file with PXR.loo.
Other learning parameters are appended to the name of the file.
}

\seealso{
\code{\link{do.RANKS}}
}

\examples{
\dontrun{
# Yeast prediction of 177 FunCat classes by leave-one-out using STRING data
# data obtained from the bionetdata package from CRAN. 
# See the AUC and Precision/recall results in the Results directory
library(bionetdata);
if (!dir.exists("data"))
  dir.create("data");
if (!dir.exists("Results"))
  dir.create("Results");
data(Yeast.STRING.data);
data(Yeast.STRING.FunCat);
save(Yeast.STRING.data, file="data/net.rda");
save(Yeast.STRING.FunCat, file="data/labels.rda");
do.loo.RANKS(data="net", labels="labels", output.name="Yeast.loo");

# another  leave-one-out prediction using KNN score and 2 steps random walk kernel
do.loo.RANKS(score = KNN.score, k=3, p=2, data="net", 
             labels="labels", output.name="Yeast.loo");
}
}

