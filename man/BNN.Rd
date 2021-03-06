% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/quickbnnr.R
\name{BNN}
\alias{BNN}
\title{Create the actual BNN model}
\usage{
BNN(
  net,
  y,
  x,
  likelihood = c("normal", "normal_seq_to_one", "tdist", "tdist_seq_to_one"),
  ...
)
}
\arguments{
\item{net}{A BNN network created usign \code{\link{make_net}}}

\item{y}{a numerical vector of length N}

\item{x}{a numerical matrix with each column being one input and one row being one variable
For example, for a network with one input, N = 100 observations, nrow(x) = 1, ncol(x) = N}

\item{likelihood}{determine the likelihood term of the model. See QuickBNN.jl for implemented likelihoods}
}
\description{
This function follows after \code{\link{make_net}} and supplies the
BNN with the data and precompiles it
}
