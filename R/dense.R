#' Create a Dense layer that has a standard normal prior for weights and biases
#'
#' @param in_size input size
#' @param out_size output size
#' @param activation activation function
#'
#' @export
DenseBNN <- function(in_size, out_size,
                     activation = c("identity", "sigmoid", "tanh", "relu")){
  activation <- match.arg(activation)
  juliacode <- sprintf("DenseBNN(%i, %i, :%s)", in_size, out_size, activation)
  return(list(in_size = in_size, out_size = out_size, activation = activation,
              julia = juliacode))
}

#' Create a Dense layer with ordered biases
#'
#' Priors for weights are standard normal. The ordering of biases is
#' implemented by putting a standard normal prior on the first bias,
#' and a truncated standard normal prior (0, inf) on the other biases.
#' The actual biases are then the cumulative sum of the biases.
#'
#' @param in_size input size
#' @param out_size output size
#' @param activation activation function
#'
#' @export
DenseOrderedBias <- function(in_size, out_size,
                             activation = c("identity", "sigmoid", "tanh", "relu")){
  activation <- match.arg(activation)
  juliacode <- sprintf("DenseOrderedBias(%i, %i, :%s)", in_size, out_size, activation)
  return(list(in_size = in_size, out_size = out_size, activation = activation,
              julia = juliacode))
}

#' Create a dense layer with ordered first weight column
#'
#' Priors for biases and all weights in columns not being the first are
#' standard normal. The first column of the weight matrix is specified
#' by putting a standard normal prior on W(1, 1) and a truncated standard normal(0, inf)
#' on the remaining weights in that column. The weights in the column are then the cumsum
#' of those prior weights
#'
#' @param in_size input size
#' @param out_size output size
#' @param activation activation function
#'
#' @export
DenseOrderedWeights <- function(in_size, out_size,
                                activation = c("identity", "sigmoid", "tanh", "relu")){
  juliacode <- sprintf("DenseOrderedWeights(%i, %i, :%s)", in_size, out_size, activation)
  return(list(in_size = in_size, out_size = out_size, activation = activation,
              julia = juliacode))
}

#' Create a layer that enforces positive weights in the first column of the weight matrix.
#' This often solves some identification issues.
#'
#' Priors for all biases and weights not in the first weight matrix column are standard normal.
#' Priors for the weights in the first column are truncted standard normal (0, inf).
#'
#' @param in_size input size
#' @param out_size output size
#' @param activation activation function
#'
#' @export
DenseForcePosFirstWeight <- function(in_size, out_size,
                                     activation = c("identity", "sigmoid", "tanh", "relu")){
  juliacode <- sprintf("DenseForcePosFirstWeight(%i, %i, :%s)", in_size, out_size, activation)
  return(list(in_size = in_size, out_size = out_size, activation = activation,
              julia = juliacode))
}

#' Create a network by chaining layers
#'
#' @param ... Layers separated by a comma
#'
#' @return Returns the julia specification of the network as a string
#'         to be passed on to \code{\link{make_net}}
#'
#' @examples
#' net <- Chain(DenseForcePosFirstWeight(1, 1, "sigmoid"), DenseBNN(1, 1))
#'
#' @export
Chain <- function(...){
  julia <- "ChainBNN("
  for (elem in list(...)){
    julia <- paste0(julia, elem$julia, ",")
  }
  julia <- paste0(julia, ")")
  return(julia)
}
