y <- arima.sim(list(ar = 0.5), n = 500)
x <- matrix(y[1:499], nrow = 1)
y <- y[2:500]

library(quickbnnr)
quickbnnr_setup()
net <- Chain(DenseForcePosFirstWeight(1, 1, "sigmoid"), DenseBNN(1, 1))
net <- make_net(net)
model <- BNN(net, y, x)
chains <- estimate(model, niter = 10000, nchains = 4)
summary(chains)

# lets get the parameters in a nicer to look at version
cleanchain <- clean_parameters(chains)
summary(cleanchain)

library(bayesplot)
mcmc_trace(chains$draws)
mcmc_dens_overlay(chains$draws)
mcmc_dens_overlay(cleanchain$draws)
mcmc_acf(chains$draws)
mcmc_intervals(chains$draws)
mcmc_pairs(chains$draws)


preds <- predict(chains)

dim(preds)
yhat <- apply(preds, 3, mean)
length(yhat)

# clearly, the theoretical mean is well matched, so is the data
plot(1:length(y), y, "l")
lines(1:length(y), yhat, col = "red")
lines(1:length(y), 0.5*x[1,], col = "green")

# this should be around 1
pp_sd <- apply(preds, 3, sd)
plot(1:length(y), pp_sd, "l", col = "red", ylim=c(min(min(pp_sd) - 0.1, 0.9), max(pp_sd) + 0.1))
abline(a = 1, b = 0, col = "black")
