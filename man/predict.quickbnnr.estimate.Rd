% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/quickbnnr.R
\name{predict.quickbnnr.estimate}
\alias{predict.quickbnnr.estimate}
\title{Uses posterior draws for prediction}
\usage{
\method{predict}{quickbnnr.estimate}(est, ...)
}
\arguments{
\item{est}{Eestimated model using \code{\link{estimate}}}

\item{...}{if x is provided, then that will be used for making predictions}
}
\description{
If a x is provided, then that x will be used for predictions,
otherwise the input to the original model will be used.
}
