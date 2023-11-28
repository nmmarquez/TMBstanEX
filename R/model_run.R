rm(list =ls())
library(TMB)
library(tmbstan)
library(INLA)
library(boot)
library(bayesplot)
beta <- 2
ar_rho <- .99
ar_sigma <- 1
obs_N <- 180
pred_N <- 200

set.seed(123)
# crude ar1
kappa <- rep(0, obs_N)
for(k in 1:obs_N){
    if(k == 1){
        kappa[k] <- rnorm(1, 0, ar_sigma)
    }else{
        kappa[k] <- rnorm(1, kappa[k-1]*ar_rho, ar_sigma)
    }
}
theta <- beta + kappa
Y_t <- rnorm(obs_N, mean = theta, sd = 1)

Data <- list(
    Y_t = as.array(Y_t)
)

DataDF <- data.frame(y=Y_t, time = 1:obs_N)

inlaRes <- inla(y~1+f(time,model="ar1"),family="gaussian", data=DataDF)

Params <- list(
    beta = 0,
    log_sigma_theta = 0,
    kappa_t = as.array(rep(0, pred_N)),
    log_sigma_ar1 = 0,
    logit_rho_ar1 = 0)

compile("./src/ar1ex.cpp")
dyn.load(dynlib("./src/ar1ex"))
Obj <- MakeADFun(
    data=Data, parameters=Params, DLL="ar1ex",
    random = "kappa_t",
    checkParameterOrder = TRUE)

# Run with just TMB
print(system.time(OptTMB <- nlminb(
    start=Obj$par, objective=Obj$fn,
    gradient=Obj$gr, control=list(eval.max=1e4, iter.max=1e4))))

sdrep <- sdreport(Obj, getJointPrecision = T)
sdrep
# how do the results compare to inla?
# should be similar even though the priors/hyperpriors were slightly different
summary(inlaRes)
c(
    "(Intercept)" = unname(OptTMB$par["beta"]),
    "Precision for the Gaussian observations" = unname(exp(
        OptTMB$par["log_sigma_theta"])^-2),
    "Precision for time" = unname(exp(OptTMB$par["log_sigma_ar1"])^-2),
    "Rho for time" = unname(inv.logit(OptTMB$par["logit_rho_ar1"])))

# run with just stan these take so long to get together
print(system.time(OptStan <- tmbstan(
     Obj, chains=3, iter = 100000)))

mcmc_intervals(
    OptStan,
    pars = c("beta", "log_sigma_theta", "log_sigma_ar1", "logit_rho_ar1"),
    transformations = c(
        "beta" = identity,
        "log_sigma_theta" = function(x) exp(x)^-2,
        "log_sigma_ar1"  = function(x) exp(x)^-2,
        "logit_rho_ar1" = inv.logit))

# run with stan and laplace approx
print(system.time(OptComb <- tmbstan(
    Obj, chains=3, iter = 100000, init = 0,
    control = list(max_treedepth = 20))))

mcmc_intervals(
    OptComb,
    pars = c("beta", "log_sigma_theta", "log_sigma_ar1", "logit_rho_ar1"),
    transformations = c(
        "beta" = identity,
        "log_sigma_theta" = function(x) exp(x)^-2,
        "log_sigma_ar1"  = function(x) exp(x)^-2,
        "logit_rho_ar1" = inv.logit))

