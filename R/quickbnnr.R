
#' Set up of the Julia environment needed for QuickBNN
#'
#' @param pkg_check Booloan; Whether to check if packages are isntalled; Default is TRUE
#' @param nthreads Int; Number of threads to use; Default is 4;
#' @param ... addtitional params passed to JuliaCall::julia_setup
#'
#' @return Nothing
#' @export
quickbnnr_setup <- function(pkg_check = TRUE, nthreads = 4, ...) {
  Sys.setenv(JULIA_NUM_THREADS = sprintf("%i", nthreads))
  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)
  if (pkg_check) {
    JuliaCall::julia_install_package_if_needed("git@github.com:enweg/QuickBNN.git")
    JuliaCall::julia_install_package_if_needed("Turing")
    JuliaCall::julia_install_package_if_needed("Flux")
  }
  JuliaCall::julia_library("QuickBNN")
  JuliaCall::julia_library("Flux")
  JuliaCall::julia_library("Turing")
}

#' Create a BNN based on a specification
#'
#' Specification must be a valid ChainBNN description from the
#' QuickBNN.jl package. This can be created using the \code{\link{Chain}}
#' function and any of the provided layer types, such as \code{\link{DenseBNN}}
#'
#' @param specification QuickBNN.jl ChainBNN specification; Can be created using \code{\link{Chain}}
#'
#' @examples
#' y <- arima.sim(list(ar = 0.5), n = 500)
#' x <- matrix(y[1:499], nrow = 1)
#' y <- y[2:500]
#' quickbnnr_setup()
#' net <- Chain(DenseBNN(1, 1, "sigmoid"), DenseBNN(1, 1))
#' net <- make_net(net)
#' model <- BNN(net, y, x)
#' chains <- estimate(model, niter = 100, nchains = 4)
#'
#' # library(bayesplot)
#' # mcmc_intervals(chains$draws)
#'
#' @export
make_net <- function(specification){
  netname = get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", netname, specification))
  out <- list(juliavar = netname,
              specification = specification)
  return(out)
}


#' Create the actual BNN model
#'
#' This function follows after \code{\link{make_net}} and supplies the
#' BNN with the data and precompiles it
#'
#' @param net A BNN network created usign \code{\link{make_net}}
#' @param y a numerical vector of length N
#' @param x a numerical matrix with each column being one input and one row being one variable
#'          For example, for a network with one input, N = 100 observations, nrow(x) = 1, ncol(x) = N
#'
#' @examples
#' y <- arima.sim(list(ar = 0.5), n = 500)
#' x <- matrix(y[1:499], nrow = 1)
#' y <- y[2:500]
#' quickbnnr_setup()
#' net <- Chain(DenseBNN(1, 1, "sigmoid"), DenseBNN(1, 1))
#' net <- make_net(net)
#' model <- BNN(net, y, x)
#' chains <- estimate(model, niter = 100, nchains = 4)
#'
#' # library(bayesplot)
#' # mcmc_intervals(chains$draws)
#'
#' @export
BNN <- function(net, y, x) {
  netname <- net$juliavar
  # making the BNN
  bnnname <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = BNN(%s);", bnnname, netname))
  # adding data to the model
  JuliaCall::julia_assign("y", y)
  JuliaCall::julia_assign("x", x)
  modelname <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s(y, x);", modelname, bnnname))

  out <- list(juliavar = modelname,
              y = y,
              x = x)
  return(out)
}

#' Estimating a BNN model using the NUTS default sampler in Turing
#'
#' @param model model formed using \code{\link{BNN}}
#' @param niter Number of MCMC iterations
#' @param nchains Number of Chains
#'
#' @return Returns a quickbnnr.estimate object with the number of iterations run, the number of chains,
#'         the draws in a format that it can be used with bayesplot, and some additional internal information
#' @export
estimate <- function(model, niter=100, nchains=1){
  model <- model$juliavar
  chainname = get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = sample(%s, NUTS(), MCMCThreads(), %i, %i)", chainname, model, niter, nchains))
  JuliaCall::julia_command(sprintf("%s = MCMCChains.get_sections(%s, :parameters)", chainname, chainname))
  varinfo <- JuliaCall::julia_eval(sprintf("String.(%s.name_map.parameters)", chainname))
  draws <- JuliaCall::julia_eval(sprintf("%s.value.data", chainname))
  # Bringing the chain into a standard format so that bayesplot can be used
  draws <- aperm(draws, c(1, 3, 2))
  dimnames(draws) <- list(iterations = NULL,
                          chains = paste0("chain:", 1:nchains),
                          parameters = varinfo)
  out <- list(juliavar = chainname,
              varinfo = varinfo,
              draws = draws,
              niter = niter,
              nchains = nchains)
  class(out) <- "quickbnnr.estimate"
  return(out)
}
