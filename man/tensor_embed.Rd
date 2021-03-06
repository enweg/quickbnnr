% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{tensor_embed}
\alias{tensor_embed}
\title{Make a tensor of sequences}
\usage{
tensor_embed(..., len_seq)
}
\arguments{
\item{...}{sequences}

\item{len_seq}{the desired sequence length}
}
\description{
Given sequences, these are usually feature sequences, tranform them
into a tensor of dimension features x num_sequences x len_sequence.
This can then be forwarded to QuickBNN.jl
}
\examples{
tensor_embed(1:5, len_seq = 2)
tensor_embed(1:5, 6:10, 11:15, len_seq = 2)

}
