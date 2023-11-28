#define TMB_LIB_INIT R_init_ar1ex
#include <TMB.hpp>
#include <Eigen/Sparse>
#include <vector>
using namespace density;
using Eigen::SparseMatrix;

// convenience function for iid random effects precision matrix
template<class Type>
SparseMatrix<Type> iid_Q(int N, Type sigma){
    SparseMatrix<Type> Q(N, N);
    for(int i = 0; i < N; i++){
        Q.insert(i,i) = 1./pow(sigma, 2.);
    }
    return Q;
}

// convenience function for ar1 random effects precision matrix
template<class Type>
SparseMatrix<Type> ar_Q(int N, Type rho, Type sigma) {
    SparseMatrix<Type> Q(N,N);
    Q.insert(0,0) = (1.) / pow(sigma, 2.);
    for (int n = 1; n < N; n++) {
        Q.insert(n,n) = (1. + pow(rho, 2.)) / pow(sigma, 2.);
        Q.insert(n-1,n) = (-1. * rho) / pow(sigma, 2.);
        Q.insert(n,n-1) = (-1. * rho) / pow(sigma, 2.);
    }
    Q.coeffRef(N-1,N-1) = (1.) / pow(sigma, 2.);
    return Q;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace R_inla;
    using namespace density;
    using namespace Eigen;

    // Observed values
    DATA_ARRAY(Y_t);

    // Parameters
    PARAMETER(beta);               // mean of time series
    PARAMETER_ARRAY(kappa_t);      // ar process
    PARAMETER(log_sigma_ar1);      // sigma of ar process
    PARAMETER(logit_rho_ar1);      // rho fo ar process
    PARAMETER(log_sigma_theta);    // sigma for observations

    // dimensions
    int T1 = Y_t.dim(0);           // number of observed time periods
    int T2 = kappa_t.dim(0);       // number of predicted time periods

    // transformed parameters
    Type sigma_ar1 = exp(log_sigma_ar1);
    Type sigma_theta = exp(log_sigma_theta);
    Type rho_ar1 = Type(1.) / (Type(1.) + exp(Type(-1.) * logit_rho_ar1));

    // define the precision matrix for the ar1 process
    SparseMatrix<Type> Q = ar_Q(T2, rho_ar1, sigma_ar1);

    // log likelihood
    Type nll = 0.0;

    // evaluate hyper priors
    nll -= dnorm(log_sigma_theta, Type(0.0), Type(10.0), true);
    nll -= dnorm(log_sigma_ar1, Type(0.0), Type(10.0), true);
    nll -= dnorm(logit_rho_ar1, Type(0.0), Type(10.0), true);

    // evaluate priors
    nll += GMRF(Q)(kappa_t);
    // you can also run the model this way using built in ar1
    //nll += SCALE(AR1(rho_ar1),sigma_ar1)(kappa_t);
    nll -= dnorm(beta, Type(0.0), Type(10.0), true);

    // evaluate data likelihood
    for(int t=0; t<T1; t++){
        nll -= dnorm(Y_t(t), beta + kappa_t(t), sigma_theta, true);
    }

    return nll;
}
