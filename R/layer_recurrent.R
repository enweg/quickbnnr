
################################################################################
#### RNN #######################################################################

#' Adds a Baysian RNN layer to the network
#'
#' @param in_size input size of the layer
#' @param out_size output size of the layer (i.e. the size
#'                 of the hidden state)
#' @param activation activation function; Standard is tanh or sigmoid
#'
#' @export
BRNN <- function(in_size, out_size,
                   activation = c("tanh", "sigmoid", "identity", "relu")){
  activation <- match.arg(activation)
  juliacode <- sprintf("BRNN(%i, %i, :%s)", in_size, out_size, activation)
  return(list(in_size = in_size, out_size = out_size, activation = activation,
              julia = juliacode))
}
