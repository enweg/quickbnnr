
#' Set up of the Julia environment needed for QuickBNN
#'
#' @param pkg_check Booloan; Whether to check if packages are isntalled; Default is TRUE
#' @param nthreads Int; Number of threads to use; Default is 4;
#' @param seed seed to be used
#' @param ... addtitional params passed to JuliaCall::julia_setup
#'
#' @return Nothing
#' @export
quickbnnr_setup <- function(pkg_check = TRUE, nthreads = 4, seed = NULL, ...) {
  Sys.setenv(JULIA_NUM_THREADS = sprintf("%i", nthreads))
  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)
  if (pkg_check) {
    JuliaCall::julia_install_package_if_needed("git@github.com:enweg/QuickBNN.git")
    JuliaCall::julia_install_package_if_needed("Turing")
    JuliaCall::julia_install_package_if_needed("Flux")
    JuliaCall::julia_install_package_if_needed("ReverseDiff")
    JuliaCall::julia_install_package_if_needed("Memoization")
    JuliaCall::julia_install_package_if_needed("Random")
  }
  JuliaCall::julia_library("QuickBNN")
  JuliaCall::julia_library("Flux")
  JuliaCall::julia_library("Turing")
  JuliaCall::julia_library("ReverseDiff")
  JuliaCall::julia_library("Memoization")
  JuliaCall::julia_library("Random")
  JuliaCall::julia_command("Turing.setadbackend(:reversediff);")
  JuliaCall::julia_command("Turing.setrdcache(true);")
  if (!is.null(seed)) quickbnnr_seed(seed)
}

#' Sets a seed for replication purposes
#'
#' @param seed seed to be used
#'
#' @export
quickbnnr_seed <- function(seed){
  JuliaCall::julia_command(sprintf("Random.seed!(%i)",
                                   seed))
  set.seed(seed)
  message(sprintf("Set the seed in both Julia and R to %i",
                  seed))
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
#' @param likelihood determine the likelihood term of the model. See QuickBNN.jl for implemented likelihoods
#'
#'
#' @export
BNN <- function(net, y, x, likelihood = c("normal", "normal_seq_to_one",
                                          "tdist", "tdist_seq_to_one"), ...) {
  likelihood <- match.arg(likelihood)
  netname <- net$juliavar
  # making the BNN
  bnnname <- get_random_symbol()
  args <- list(...)
  if ("nu" %in% names(args) & substr(likelihood, 1, 5) == "tdist"){
    nuname <- get_random_symbol()
    JuliaCall::julia_assign(nuname, args[["nu"]])
    JuliaCall::julia_eval(nuname)
    JuliaCall::julia_command(sprintf("%s = BNN(%s, QuickBNN.likelihood_%s; ν = %s);",
                                     bnnname, netname, likelihood, nuname))
  }
  else {
    JuliaCall::julia_command(sprintf("%s = BNN(%s, QuickBNN.likelihood_%s);",
                                     bnnname, netname, likelihood))
  }
  # adding data to the model
  JuliaCall::julia_assign("y", y)
  JuliaCall::julia_assign("x", x)
  if (ndims(x) == 3){
    # RNN case
    JuliaCall::julia_command("x = to_RNN_format(x);")
  }
  modelname <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s(y, x);", modelname, bnnname))

  out <- list(juliavar = list(model = modelname, bnn = bnnname),
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
  mod <- model
  model <- model$juliavar$model
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
              nchains = nchains,
              model = mod)
  class(out) <- "quickbnnr.estimate"
  return(out)
}

#' Clean parameters for better understanding
#'
#' Some of the layers implemented take cumsums of variables to obtain
#' the actual NN parameters used. \code{\link{estimate}} will return
#' the parameters before taking the cumsum and thus can be difficult
#' to interpret. This method fixes this.
#'
#' @param est estimated model using \code{\link{estimate}}
#'
#' @export
clean_parameters <- function(est){
  cleanchain <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = generated_quantities_chain(%s, %s)",
                                   cleanchain, est$model$juliavar$model, est$juliavar))
  varinfo <- JuliaCall::julia_eval(sprintf("String.(%s.name_map.parameters)", cleanchain))
  draws <- JuliaCall::julia_eval(sprintf("%s.value.data", cleanchain))
  draws <- aperm(draws, c(1, 3, 2))
  dimnames(draws) <- list(iterations = NULL,
                          chains = paste0("chain:", 1:est$nchains),
                          parameters = varinfo)
  out <- list(juliavar = cleanchain,
              varinfo = varinfo,
              draws = draws,
              niter = est$niter,
              nchains = est$nchains,
              model = est$model)
  class(out) <- "quickbnnr.estimate"
  return(out)
}


#' Uses posterior draws for prediction
#'
#' If a x is provided, then that x will be used for predictions,
#' otherwise the input to the original model will be used.
#'
#' @param est Eestimated model using \code{\link{estimate}}
#' @param ... if x is provided, then that will be used for making predictions
#'
#' @export
predict.quickbnnr.estimate <- function(est, ...){
  args <- list(...)
  if ('x' %in% names(args)) x <- args$x else x <- est$model$x
  predictname <- get_random_symbol()
  JuliaCall::julia_assign(sprintf("%s_x", predictname), x)
  if (ndims(x) == 3) JuliaCall::julia_command(sprintf("%s_x = to_RNN_format(%s_x)",
                                                      predictname, predictname))
  JuliaCall::julia_command(sprintf("%s = posterior_predictive(%s, %s_x, %s)",
                                   predictname, est$model$juliavar$bnn, predictname, est$juliavar))
  varinfo <- JuliaCall::julia_eval(sprintf("String.(%s.name_map.parameters)", predictname))
  draws <- JuliaCall::julia_eval(sprintf("%s.value.data", predictname))
  # Bringing this into a standard format
  draws <- aperm(draws, c(1, 3, 2))
  dimnames(draws) <- list(iterations = NULL,
                          chains = paste0("chain:", 1:est$nchains),
                          parameters = varinfo)
  return(draws)
}


#' Just a helper function
.summary <- function(summaryvar, section, to_string = FALSE){
  if (to_string) {
    return (JuliaCall::julia_eval(sprintf("String.(%s.nt.%s)", summaryvar, section)))
  }
  JuliaCall::julia_eval(sprintf("%s.nt.%s", summaryvar, section))
}

#' Returns summary statistics of an estimated model
#'
#' @param est model estimated using \code{\link{estimate}}
#'
#' @export
summary.quickbnnr.estimate <- function(est){
  summaryvar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = describe(%s)[1];", summaryvar, est$juliavar))
  ess <- .summary(summaryvar, "ess")
  ess_per_sec <- .summary(summaryvar, "ess_per_sec")
  mcse <- .summary(summaryvar, "mcse")
  mu <- .summary(summaryvar, "mean")
  naive_se <- .summary(summaryvar, "naive_se")
  parameters <- .summary(summaryvar, "parameters", to_string = TRUE)
  rhat <- .summary(summaryvar, "rhat")
  std <- .summary(summaryvar, "std")
  return(data.frame(list(
    Parameter = parameters,
    mean = mu,
    std = std,
    naive_se = naive_se,
    mcse = mcse,
    ess = ess,
    ess_per_sec = ess_per_sec,
    rhat = rhat
  )))
}

