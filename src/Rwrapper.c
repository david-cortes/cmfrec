/*******************************************************************************
    Collective Matrix Factorization
    -------------------------------
    
    This is a module for multi-way factorization of sparse and dense matrices
    intended to be used for recommender system with explicit feedback data plus
    side information about users and/or items.

    The reference papers are:
        (a) Cortes, David.
            "Cold-start recommendations in Collective Matrix Factorization."
            arXiv preprint arXiv:1809.00366 (2018).
        (b) Singh, Ajit P., and Geoffrey J. Gordon.
            "Relational learning via collective matrix factorization."
            Proceedings of the 14th ACM SIGKDD international conference on
            Knowledge discovery and data mining. 2008.
        (c) Hu, Yifan, Yehuda Koren, and Chris Volinsky.
            "Collaborative filtering for implicit feedback datasets."
            2008 Eighth IEEE International Conference on Data Mining.
            Ieee, 2008.
        (d) Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk.
            "Applications of the conjugate gradient method for
            implicit feedback collaborative filtering."
            Proceedings of the fifth ACM conference on
            Recommender systems. 2011.
        (e) Rendle, Steffen, Li Zhang, and Yehuda Koren.
            "On the difficulty of evaluating baselines:
            A study on recommender systems."
            arXiv preprint arXiv:1905.01395 (2019).
        (f) Franc, Vojtech, Vaclav Hlavac, and Mirko Navara.
            "Sequential coordinate-wise algorithm for the
            non-negative least squares problem."
            International Conference on Computer Analysis of Images
            and Patterns. Springer, Berlin, Heidelberg, 2005.
        (e) Zhou, Yunhong, et al.
            "Large-scale parallel collaborative filtering for the netflix prize."
            International conference on algorithmic applications in management.
            Springer, Berlin, Heidelberg, 2008.

    For information about the models offered here and how they are fit to
    the data, see the files 'collective.c' and 'offsets.c'.

    Written for C99 standard and OpenMP version 2.0 or higher, and aimed to be
    used either as a stand-alone program, or wrapped into scripting languages
    such as Python and R.
    <https://www.github.com/david-cortes/cmfrec>

    

    MIT License:

    Copyright (c) 2021 David Cortes

    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

*******************************************************************************/
#ifdef _FOR_R

#include "cmfrec.h"

SEXP deep_copy(SEXP x)
{
    SEXP out = PROTECT(Rf_allocVector(REALSXP, Rf_xlength(x)));
    if (Rf_xlength(x))
        memcpy(REAL(out), REAL(x), (size_t)Rf_xlength(x)*sizeof(double));
    UNPROTECT(1);
    return out;
}

SEXP as_size_t(SEXP x)
{
    size_t n = (size_t)Rf_xlength(x);
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n*sizeof(size_t)));
    int *ptr_x = INTEGER(x);
    size_t *ptr_out = (size_t*)RAW(out);
    for (size_t ix = 0; ix < n; ix++)
        ptr_out[ix] = ptr_x[ix];
    UNPROTECT(1);
    return out;
}

double* get_ptr(SEXP x)
{
    if (Rf_xlength(x))
        return REAL(x);
    else
        return (double*)NULL;
}

int* get_ptr_int(SEXP x)
{
    if (Rf_xlength(x))
        return INTEGER(x);
    else
        return (int*)NULL;
}

size_t* get_ptr_size_t(SEXP x)
{
    if (Rf_xlength(x))
        return (size_t*) RAW(x);
    else
        return (size_t*)NULL;
}

/* ---------------------------------------------------- */

SEXP call_fit_collective_explicit_lbfgs
(
    SEXP biasA, SEXP biasB,
    SEXP A, SEXP B,
    SEXP C, SEXP Cb,
    SEXP D, SEXP Db,
    SEXP glob_mean,
    SEXP U_colmeans, SEXP I_colmeans,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull, SEXP Wsp,
    SEXP user_bias, SEXP item_bias, SEXP center,
    SEXP lam,
    SEXP U, SEXP m_u, SEXP p,
    SEXP II, SEXP n_i, SEXP q,
    SEXP Ub, SEXP m_ubin, SEXP pbin,
    SEXP Ib, SEXP n_ibin, SEXP qbin,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP k_main, SEXP k_user, SEXP k_item,
    SEXP w_main, SEXP w_user, SEXP w_item,
    SEXP n_corr_pairs, SEXP maxiter, SEXP print_every,
    SEXP nupd, SEXP nfev,
    SEXP prefer_onepass,
    SEXP nthreads, SEXP verbose, SEXP handle_interrupt,
    SEXP precompute,
    SEXP include_all_X,
    SEXP B_plus_bias,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BeTBeChol,
    SEXP TransCtCinvCt,
    SEXP CtCw
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = fit_collective_explicit_lbfgs(
        get_ptr(biasA), get_ptr(biasB),
        REAL(A), REAL(B),
        get_ptr(C), get_ptr(Cb),
        get_ptr(D), get_ptr(Db),
        true, 1,
        REAL(glob_mean),
        get_ptr(U_colmeans), get_ptr(I_colmeans),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t)Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(user_bias), (bool) Rf_asLogical(item_bias),
        Rf_asLogical(center),
        lambda_, lam_unique,
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(n_i), Rf_asInteger(q),
        get_ptr(Ub), Rf_asInteger(m_ubin), Rf_asInteger(pbin),
        get_ptr(Ib), Rf_asInteger(n_ibin), Rf_asInteger(qbin),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t)Rf_xlength(U_sp),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t)Rf_xlength(I_sp),
        Rf_asInteger(k_main), Rf_asInteger(k_user), Rf_asInteger(k_item),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_item),
        Rf_asInteger(n_corr_pairs), (size_t) Rf_asInteger(maxiter),
        Rf_asInteger(nthreads), (bool) Rf_asLogical(prefer_onepass),
        (bool) Rf_asLogical(verbose), Rf_asInteger(print_every),
        (bool) Rf_asLogical(handle_interrupt),
        INTEGER(nupd), INTEGER(nfev),
        (bool) Rf_asLogical(precompute),
        (bool) Rf_asLogical(include_all_X),
        get_ptr(B_plus_bias),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BeTBeChol),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out;
}

/* Note: R's '.Call' has a limit of 65 arguments, thus some need to be merged */
SEXP call_fit_collective_explicit_als
(
    SEXP biasA, SEXP biasB,
    SEXP A, SEXP B,
    SEXP C, SEXP D,
    SEXP Ai, SEXP Bi,
    SEXP glob_mean,
    SEXP U_colmeans, SEXP I_colmeans,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull, SEXP Wsp,
    SEXP user_bias, SEXP item_bias, SEXP center,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP U, SEXP m_u, SEXP p,
    SEXP II, SEXP n_i, SEXP q,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP NA_as_zero_X, SEXP NA_as_zero_U, SEXP NA_as_zero_I,
    SEXP k_main_k_user_k_item,
    SEXP w_main_w_user_w_item_w_implicit,
    SEXP niter, SEXP nthreads, SEXP verbose, SEXP handle_interrupt,
    SEXP use_cg, SEXP max_cg_steps, SEXP finalize_chol,
    SEXP nonneg, SEXP max_cd_steps, SEXP nonneg_CD,
    SEXP precompute,
    SEXP add_implicit_features,
    SEXP include_all_X,
    SEXP B_plus_bias,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP TransCtCinvCt,
    SEXP CtCw
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);

    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);


    bool nonneg_C = (bool) LOGICAL(nonneg_CD)[0];
    bool nonneg_D = (bool) LOGICAL(nonneg_CD)[1];

    double w_main = REAL(w_main_w_user_w_item_w_implicit)[0];
    double w_user = REAL(w_main_w_user_w_item_w_implicit)[1];
    double w_item = REAL(w_main_w_user_w_item_w_implicit)[2];
    double w_implicit = REAL(w_main_w_user_w_item_w_implicit)[3];

    int k_main = INTEGER(k_main_k_user_k_item)[0];
    int k_user = INTEGER(k_main_k_user_k_item)[1];
    int k_item = INTEGER(k_main_k_user_k_item)[2];

    int retval = fit_collective_explicit_als(
        get_ptr(biasA), get_ptr(biasB),
        REAL(A), REAL(B),
        get_ptr(C), get_ptr(D),
        get_ptr(Ai), get_ptr(Bi),
        (bool) Rf_asLogical(add_implicit_features),
        true, 1,
        REAL(glob_mean),
        get_ptr(U_colmeans), get_ptr(I_colmeans),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t)Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(user_bias), (bool) Rf_asLogical(item_bias),
        (bool) Rf_asLogical(center),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(n_i), Rf_asInteger(q),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t)Rf_xlength(U_sp),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t)Rf_xlength(I_sp),
        (bool) Rf_asLogical(NA_as_zero_X), (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_I),
        k_main, k_user, k_item,
        w_main, w_user, w_item, w_implicit,
        Rf_asInteger(niter), Rf_asInteger(nthreads), (bool) Rf_asLogical(verbose),
        (bool) Rf_asLogical(handle_interrupt),
        (bool) Rf_asLogical(use_cg), Rf_asInteger(max_cg_steps), (bool) Rf_asLogical(finalize_chol),
        (bool) Rf_asLogical(nonneg), (size_t) Rf_asInteger(max_cd_steps),
        nonneg_C, nonneg_D,
        (bool) Rf_asLogical(precompute),
        (bool) Rf_asLogical(include_all_X),
        get_ptr(B_plus_bias),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out;
}

SEXP call_fit_collective_implicit_als
(
    SEXP A, SEXP B,
    SEXP C, SEXP D,
    SEXP w_main_multiplier,
    SEXP U_colmeans, SEXP I_colmeans,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP lam, SEXP l1_lam,
    SEXP U, SEXP m_u, SEXP p,
    SEXP II, SEXP n_i, SEXP q,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP NA_as_zero_U, SEXP NA_as_zero_I,
    SEXP k_main, SEXP k_user, SEXP k_item,
    SEXP w_main, SEXP w_user, SEXP w_item,
    SEXP niter, SEXP nthreads, SEXP verbose, SEXP handle_interrupt,
    SEXP use_cg, SEXP max_cg_steps, SEXP finalize_chol,
    SEXP nonneg, SEXP max_cd_steps, SEXP nonneg_C, SEXP nonneg_D,
    SEXP alpha, SEXP adjust_weight, SEXP apply_log_transf,
    SEXP precompute,
    SEXP BtB,
    SEXP BeTBe,
    SEXP BeTBeChol
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);

    REAL(w_main_multiplier)[0] = 1.;


    int retval = fit_collective_implicit_als(
        REAL(A), REAL(B),
        get_ptr(C), get_ptr(D),
        true, 1,
        get_ptr(U_colmeans), get_ptr(I_colmeans),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t)Rf_xlength(X),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(n_i), Rf_asInteger(q),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t)Rf_xlength(U_sp),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t)Rf_xlength(I_sp),
        (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_I),
        Rf_asInteger(k_main), Rf_asInteger(k_user), Rf_asInteger(k_item),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_item),
        REAL(w_main_multiplier),
        Rf_asReal(alpha), (bool) Rf_asLogical(adjust_weight),
        (bool) Rf_asLogical(apply_log_transf),
        Rf_asInteger(niter), Rf_asInteger(nthreads), (bool) Rf_asLogical(verbose),
        (bool) Rf_asLogical(handle_interrupt),
        (bool) Rf_asLogical(use_cg), Rf_asInteger(max_cg_steps), (bool) Rf_asLogical(finalize_chol),
        (bool) Rf_asLogical(nonneg), (size_t) Rf_asInteger(max_cd_steps),
        (bool) Rf_asLogical(nonneg_C), (bool) Rf_asLogical(nonneg_D),
        (bool) Rf_asLogical(precompute),
        get_ptr(BtB),
        get_ptr(BeTBe),
        get_ptr(BeTBeChol)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out;
}


SEXP call_fit_most_popular
(
    SEXP biasA, SEXP biasB,
    SEXP glob_mean,
    SEXP lam,
    SEXP scale_lam,
    SEXP alpha,
    SEXP m, SEXP n,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull,
    SEXP Wsp,
    SEXP implicit, SEXP adjust_weight,
    SEXP apply_log_transf,
    SEXP nonneg,
    SEXP w_main_multiplier,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    REAL(w_main_multiplier)[0] = 1.;

    double lam_user = lambda_;
    double lam_item = lambda_;
    if (lam_unique != NULL) {
        lam_user = lam_unique[0];
        lam_item = lam_unique[1];
    }


    int retval = fit_most_popular(
        get_ptr(biasA), get_ptr(biasB),
        REAL(glob_mean),
        lam_user, lam_item,
        (bool) Rf_asLogical(scale_lam),
        Rf_asReal(alpha),
        Rf_asInteger(m), Rf_asInteger(n),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t)Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(implicit),
        (bool) Rf_asLogical(adjust_weight),
        (bool) Rf_asLogical(apply_log_transf),
        (bool) Rf_asLogical(nonneg),
        REAL(w_main_multiplier),
        Rf_asInteger(nthreads)
    );


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out;
}

SEXP call_fit_content_based_lbfgs
(
    SEXP biasA, SEXP biasB,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP start_with_ALS,
    SEXP glob_mean,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull,
    SEXP Wsp,
    SEXP user_bias, SEXP item_bias,
    SEXP add_intercepts,
    SEXP lam,
    SEXP U, SEXP p,
    SEXP II, SEXP q,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP n_corr_pairs, SEXP maxiter,
    SEXP nthreads, SEXP prefer_onepass,
    SEXP verbose, SEXP print_every, SEXP handle_interrupt,
    SEXP niter, SEXP nfev,
    SEXP Am, SEXP Bm
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);


    int retval = fit_content_based_lbfgs(
        get_ptr(biasA), get_ptr(biasB),
        REAL(C), get_ptr(C_bias),
        REAL(D), get_ptr(D_bias),
        (bool) Rf_asLogical(start_with_ALS),
        true, 1,
        REAL(glob_mean),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t)Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(user_bias), (bool) Rf_asLogical(item_bias),
        (bool) Rf_asLogical(add_intercepts),
        lambda_, lam_unique,
        get_ptr(U), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(q),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t)Rf_xlength(U_sp),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t)Rf_xlength(I_sp),
        Rf_asInteger(n_corr_pairs), (size_t) Rf_asInteger(maxiter),
        Rf_asInteger(nthreads), (bool) Rf_asLogical(prefer_onepass),
        (bool) Rf_asLogical(verbose), Rf_asInteger(print_every),
        (bool) Rf_asLogical(handle_interrupt),
        INTEGER(niter), INTEGER(nfev),
        REAL(Am), REAL(Bm)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out;
}


SEXP call_fit_offsets_explicit_lbfgs
(
    SEXP biasA, SEXP biasB,
    SEXP A, SEXP B,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP glob_mean,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull, SEXP Wsp,
    SEXP user_bias, SEXP item_bias, SEXP center,
    SEXP add_intercepts,
    SEXP lam,
    SEXP U, SEXP p,
    SEXP II, SEXP q,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP k_main, SEXP k_sec,
    SEXP w_user, SEXP w_item,
    SEXP n_corr_pairs, SEXP maxiter,
    SEXP nthreads, SEXP prefer_onepass,
    SEXP verbose, SEXP print_every, SEXP handle_interrupt,
    SEXP niter, SEXP nfev,
    SEXP precompute,
    SEXP Am, SEXP Bm,
    SEXP Bm_plus_bias,
    SEXP BtB,
    SEXP TransBtBinvBt
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = fit_offsets_explicit_lbfgs(
        get_ptr(biasA), get_ptr(biasB),
        get_ptr(A), get_ptr(B),
        get_ptr(C), get_ptr(C_bias),
        get_ptr(D), get_ptr(D_bias),
        true, 1,
        REAL(glob_mean),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t) Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(user_bias), (bool) Rf_asLogical(item_bias),
        (bool)Rf_asLogical(center), (bool) Rf_asLogical(add_intercepts),
        lambda_, lam_unique,
        get_ptr(U), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(q),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t) Rf_xlength(I_sp),
        Rf_asInteger(k_main), Rf_asInteger(k_sec),
        Rf_asReal(w_user), Rf_asReal(w_item),
        Rf_asInteger(n_corr_pairs), (size_t) Rf_asInteger(maxiter),
        Rf_asInteger(nthreads), (bool) Rf_asLogical(prefer_onepass),
        (bool) Rf_asLogical(verbose), Rf_asInteger(print_every),
        (bool) Rf_asLogical(handle_interrupt),
        INTEGER(niter), INTEGER(nfev),
        (bool) Rf_asLogical(precompute),
        get_ptr(Am), get_ptr(Bm),
        get_ptr(Bm_plus_bias),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}


SEXP call_fit_offsets_explicit_als
(
    SEXP biasA, SEXP biasB,
    SEXP A, SEXP B,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP glob_mean,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP Xfull,
    SEXP Wfull, SEXP Wsp,
    SEXP user_bias, SEXP item_bias, SEXP center, SEXP add_intercepts,
    SEXP lam,
    SEXP U, SEXP p,
    SEXP II, SEXP q,
    SEXP NA_as_zero_X,
    SEXP niter,
    SEXP nthreads, SEXP use_cg,
    SEXP max_cg_steps, SEXP finalize_chol,
    SEXP verbose, SEXP handle_interrupt,
    SEXP precompute,
    SEXP Am, SEXP Bm,
    SEXP Bm_plus_bias,
    SEXP BtB,
    SEXP TransBtBinvBt
)
{
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = fit_offsets_explicit_als(
        get_ptr(biasA), get_ptr(biasB),
        REAL(A), REAL(B),
        get_ptr(C), get_ptr(C_bias),
        get_ptr(D), get_ptr(D_bias),
        true, 1,
        REAL(glob_mean),
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        get_ptr_int(ixA), get_ptr_int(ixB), get_ptr(X), (size_t) Rf_xlength(X),
        get_ptr(Xfull),
        weight,
        (bool) Rf_asLogical(user_bias), (bool) Rf_asLogical(item_bias),
        (bool) Rf_asLogical(center), (bool) Rf_asLogical(add_intercepts),
        Rf_asReal(lam),
        get_ptr(U), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(q),
        (bool) Rf_asLogical(NA_as_zero_X),
        Rf_asInteger(niter),
        Rf_asInteger(nthreads), (bool) Rf_asLogical(use_cg),
        Rf_asInteger(max_cg_steps), (bool) Rf_asLogical(finalize_chol),
        (bool) Rf_asLogical(verbose), (bool) Rf_asLogical(handle_interrupt),
        (bool) Rf_asLogical(precompute),
        get_ptr(Am), get_ptr(Bm),
        get_ptr(Bm_plus_bias),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_fit_offsets_implicit_als
(
    SEXP A, SEXP B,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP m, SEXP n, SEXP k,
    SEXP ixA, SEXP ixB, SEXP X,
    SEXP add_intercepts,
    SEXP lam,
    SEXP U, SEXP p,
    SEXP II, SEXP q,
    SEXP alpha, SEXP apply_log_transf,
    SEXP niter,
    SEXP nthreads, SEXP use_cg,
    SEXP max_cg_steps, SEXP finalize_chol,
    SEXP verbose, SEXP handle_interrupt,
    SEXP precompute,
    SEXP Am, SEXP Bm,
    SEXP BtB
)
{
    int retval = fit_offsets_implicit_als(
        REAL(A), REAL(B),
        get_ptr(C), get_ptr(C_bias),
        get_ptr(D), get_ptr(D_bias),
        true, 1,
        Rf_asInteger(m), Rf_asInteger(n), Rf_asInteger(k),
        INTEGER(ixA), INTEGER(ixB), REAL(X), (size_t) Rf_xlength(X),
        (bool) Rf_asLogical(add_intercepts),
        Rf_asReal(lam),
        get_ptr(U), Rf_asInteger(p),
        get_ptr(II), Rf_asInteger(q),
        Rf_asReal(alpha), (bool) Rf_asLogical(apply_log_transf),
        Rf_asInteger(niter),
        Rf_asInteger(nthreads), (bool) Rf_asLogical(use_cg),
        Rf_asInteger(max_cg_steps), (bool) Rf_asLogical(finalize_chol),
        (bool) Rf_asLogical(verbose), (bool) Rf_asLogical(handle_interrupt),
        (bool) Rf_asLogical(precompute),
        get_ptr(Am), get_ptr(Bm),
        get_ptr(BtB)
    );

    if (!((bool) Rf_asLogical(handle_interrupt)))
        R_CheckUserInterrupt();

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_precompute_collective_explicit
(
    SEXP B, SEXP n, SEXP n_max, SEXP include_all_X,
    SEXP C, SEXP p,
    SEXP Bi, SEXP add_implicit_features,
    SEXP biasB, SEXP glob_mean, SEXP NA_as_zero_X,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP user_bias,
    SEXP nonneg,
    SEXP lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP B_plus_bias,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP TransCtCinvCt,
    SEXP CtCw
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    int retval = precompute_collective_explicit(
        REAL(B), Rf_asInteger(n), Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(C), Rf_asInteger(p),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        get_ptr(biasB), Rf_asReal(glob_mean), (bool) Rf_asLogical(NA_as_zero_X),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        (bool) Rf_asLogical(user_bias),
        (bool) Rf_asLogical(nonneg),
        lambda_, lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        get_ptr(B_plus_bias),
        REAL(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_precompute_collective_implicit
(
    SEXP B, SEXP n,
    SEXP C, SEXP p,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP w_main, SEXP w_user, SEXP w_main_multiplier,
    SEXP nonneg,
    SEXP extra_precision,
    SEXP BtB,
    SEXP BeTBe,
    SEXP BeTBeChol
)
{
    int retval = precompute_collective_implicit(
        REAL(B), Rf_asInteger(n),
        get_ptr(C), Rf_asInteger(p),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asReal(lam), Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_main_multiplier),
        (bool) Rf_asLogical(nonneg),
        (bool) Rf_asLogical(extra_precision),
        get_ptr(BtB),
        get_ptr(BeTBe),
        get_ptr(BeTBeChol)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}


/* ---------------------------------------------------- */

SEXP call_factors_collective_explicit_single
(
    SEXP a_vec, SEXP a_bias,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP u_bin_vec, SEXP pbin,
    SEXP NA_as_zero_U, SEXP NA_as_zero_X,
    SEXP nonneg,
    SEXP C, SEXP Cb,
    SEXP glob_mean, SEXP biasB,
    SEXP U_colmeans,
    SEXP Xa, SEXP ixB,
    SEXP Xa_dense, SEXP n,
    SEXP weight,
    SEXP B,
    SEXP Bi, SEXP add_implicit_features,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP n_max, SEXP include_all_X,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP CtCw,
    SEXP TransCtCinvCt,
    SEXP B_plus_bias
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);

    int retval = factors_collective_explicit_single(
        REAL(a_vec), get_ptr(a_bias),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(u_bin_vec), Rf_asInteger(pbin),
        (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_X),
        (bool) Rf_asLogical(nonneg),
        get_ptr(C), get_ptr(Cb),
        Rf_asReal(glob_mean), get_ptr(biasB),
        get_ptr(U_colmeans),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Xa_dense), Rf_asInteger(n),
        get_ptr(weight),
        get_ptr(B),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(CtCw),
        get_ptr(TransCtCinvCt),
        get_ptr(B_plus_bias)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}


SEXP call_factors_collective_implicit_single
(
    SEXP a_vec,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP NA_as_zero_U,
    SEXP nonneg,
    SEXP U_colmeans,
    SEXP B, SEXP n, SEXP C,
    SEXP Xa, SEXP ixB,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam, SEXP alpha, SEXP w_main, SEXP w_user,
    SEXP w_main_multiplier,
    SEXP apply_log_transf,
    SEXP BeTBe,
    SEXP BtB,
    SEXP BeTBeChol
)
{
    int retval = factors_collective_implicit_single(
        REAL(a_vec),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        (bool) Rf_asLogical(NA_as_zero_U),
        (bool) Rf_asLogical(nonneg),
        get_ptr(U_colmeans),
        get_ptr(B), Rf_asInteger(n), get_ptr(C),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asReal(lam), Rf_asReal(l1_lam), Rf_asReal(alpha), Rf_asReal(w_main), Rf_asReal(w_user),
        Rf_asReal(w_main_multiplier),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(BeTBe),
        get_ptr(BtB),
        get_ptr(BeTBeChol)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_content_based_single
(
    SEXP a_vec, SEXP k,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP C, SEXP C_bias
)
{
    int retval = factors_content_based_single(
        REAL(a_vec), Rf_asInteger(k),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        REAL(C), get_ptr(C_bias)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_offsets_explicit_single
(
    SEXP a_vec, SEXP a_bias, SEXP output_a,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP Xa, SEXP ixB,
    SEXP Xa_dense, SEXP n,
    SEXP weight,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP glob_mean, SEXP biasB,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP w_user,
    SEXP lam,
    SEXP exact,
    SEXP precomputedTransBtBinvBt,
    SEXP precomputedBtB,
    SEXP Bm_plus_bias
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    int retval = factors_offsets_explicit_single(
        REAL(a_vec), get_ptr(a_bias), get_ptr(output_a),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Xa_dense), Rf_asInteger(n),
        get_ptr(weight),
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asReal(glob_mean), get_ptr(biasB),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        Rf_asReal(w_user),
        lambda_, lam_unique,
        (bool) Rf_asLogical(exact),
        get_ptr(precomputedTransBtBinvBt),
        get_ptr(precomputedBtB),
        get_ptr(Bm_plus_bias)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_offsets_implicit_single
(
    SEXP a_vec,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP Xa, SEXP ixB,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP k, SEXP n,
    SEXP lam, SEXP alpha,
    SEXP apply_log_transf,
    SEXP precomputedBtB,
    SEXP output_a
)
{
    int retval = factors_offsets_implicit_single(
        REAL(a_vec),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asInteger(k), Rf_asInteger(n),
        Rf_asReal(lam), Rf_asReal(alpha),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(precomputedBtB),
        get_ptr(output_a)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_factors_collective_explicit_multiple
(
    SEXP A, SEXP biasA, SEXP m,
    SEXP U, SEXP m_u, SEXP p,
    SEXP NA_as_zero_U, SEXP NA_as_zero_X,
    SEXP nonneg,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP Ub, SEXP m_ubin, SEXP pbin,
    SEXP C, SEXP Cb,
    SEXP glob_mean, SEXP biasB,
    SEXP U_colmeans,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Xfull, SEXP n,
    SEXP Wfull, SEXP Wsp,
    SEXP B,
    SEXP Bi, SEXP add_implicit_features,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP n_max, SEXP include_all_X,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP TransCtCinvCt,
    SEXP CtCw,
    SEXP B_plus_bias,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);

    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = factors_collective_explicit_multiple(
        REAL(A), get_ptr(biasA), Rf_asInteger(m),
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_X),
        (bool) Rf_asLogical(nonneg),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_row),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(Ub), Rf_asInteger(m_ubin), Rf_asInteger(pbin),
        get_ptr(C), get_ptr(Cb),
        Rf_asReal(glob_mean), get_ptr(biasB),
        get_ptr(U_colmeans),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Xfull), Rf_asInteger(n),
        weight,
        get_ptr(B),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw),
        get_ptr(B_plus_bias),
        Rf_asInteger(nthreads)
    );


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}


SEXP call_factors_collective_implicit_multiple
(
    SEXP A, SEXP m,
    SEXP U, SEXP m_u, SEXP p,
    SEXP NA_as_zero_U,
    SEXP nonneg,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP B, SEXP n,
    SEXP C,
    SEXP U_colmeans,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam, SEXP alpha, SEXP w_main, SEXP w_user,
    SEXP w_main_multiplier,
    SEXP apply_log_transf,
    SEXP BeTBe,
    SEXP BtB,
    SEXP BeTBeChol,
    SEXP nthreads
)
{
    int retval = factors_collective_implicit_multiple(
        REAL(A), Rf_asInteger(m),
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        (bool) Rf_asLogical(NA_as_zero_U),
        (bool) Rf_asLogical(nonneg),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(B), Rf_asInteger(n),
        get_ptr(C),
        get_ptr(U_colmeans),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asReal(lam), Rf_asReal(l1_lam), Rf_asReal(alpha), Rf_asReal(w_main), Rf_asReal(w_user),
        Rf_asReal(w_main_multiplier),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(BeTBe),
        get_ptr(BtB),
        get_ptr(BeTBeChol),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_content_based_mutliple
(
    SEXP Am, SEXP m_new, SEXP k,
    SEXP C, SEXP C_bias,
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP nthreads
)
{
    int retval = factors_content_based_mutliple(
        REAL(Am), Rf_asInteger(m_new), Rf_asInteger(k),
        REAL(C), get_ptr(C_bias),
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_offsets_explicit_multiple
(
    SEXP Am, SEXP biasA,
    SEXP A, SEXP m,
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Xfull, SEXP n,
    SEXP Wfull, SEXP Wsp,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP glob_mean, SEXP biasB,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP w_user,
    SEXP lam, SEXP exact,
    SEXP precomputedTransBtBinvBt,
    SEXP precomputedBtB,
    SEXP Bm_plus_bias,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = factors_offsets_explicit_multiple(
        REAL(Am), get_ptr(biasA),
        get_ptr(A), Rf_asInteger(m),
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Xfull), Rf_asInteger(n),
        weight,
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asReal(glob_mean), get_ptr(biasB),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        Rf_asReal(w_user),
        lambda_, lam_unique, (bool) Rf_asLogical(exact),
        get_ptr(precomputedTransBtBinvBt),
        get_ptr(precomputedBtB),
        get_ptr(Bm_plus_bias),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_factors_offsets_implicit_multiple
(
    SEXP Am, SEXP m,
    SEXP A,
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP k, SEXP n,
    SEXP lam, SEXP alpha,
    SEXP apply_log_transf,
    SEXP precomputedBtB,
    SEXP nthreads
)
{
    int retval = factors_offsets_implicit_multiple(
        REAL(Am), Rf_asInteger(m),
        get_ptr(A),
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asInteger(k), Rf_asInteger(n),
        Rf_asReal(lam), Rf_asReal(alpha),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(precomputedBtB),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_impute_X_collective_explicit
(
    SEXP m, SEXP user_bias,
    SEXP U, SEXP m_u, SEXP p,
    SEXP NA_as_zero_U,
    SEXP nonneg,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP Ub, SEXP m_ubin, SEXP pbin,
    SEXP C, SEXP Cb,
    SEXP glob_mean, SEXP biasB,
    SEXP U_colmeans,
    SEXP Xfull, SEXP n,
    SEXP Wfull,
    SEXP B,
    SEXP Bi, SEXP add_implicit_features,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP n_max, SEXP include_all_X,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP TransCtCinvCt,
    SEXP CtCw,
    SEXP B_plus_bias,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);

    int retval = impute_X_collective_explicit(
        Rf_asInteger(m), (bool) Rf_asLogical(user_bias),
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        (bool) Rf_asLogical(NA_as_zero_U),
        (bool) Rf_asLogical(nonneg),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(Ub), Rf_asInteger(m_ubin), Rf_asInteger(pbin),
        get_ptr(C), get_ptr(Cb),
        Rf_asReal(glob_mean), get_ptr(biasB),
        get_ptr(U_colmeans),
        REAL(Xfull), Rf_asInteger(n),
        get_ptr(Wfull),
        get_ptr(B),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw),
        get_ptr(B_plus_bias),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_predict_X_old_collective_explicit
(
    SEXP row, SEXP col, SEXP predicted,
    SEXP A, SEXP biasA,
    SEXP B, SEXP biasB,
    SEXP glob_mean,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP m, SEXP n_max,
    SEXP nthreads
)
{
    int retval = predict_X_old_collective_explicit(
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        REAL(A), get_ptr(biasA),
        REAL(B), get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asInteger(m), Rf_asInteger(n_max),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_old_collective_implicit
(
    SEXP row, SEXP col, SEXP predicted,
    SEXP A,
    SEXP B,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP m, SEXP n,
    SEXP nthreads
)
{
    int retval = predict_X_old_collective_implicit(
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        REAL(A),
        REAL(B),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asInteger(m), Rf_asInteger(n),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_old_most_popular
(
    SEXP row, SEXP col, SEXP predicted,
    SEXP biasA, SEXP biasB,
    SEXP glob_mean,
    SEXP m, SEXP n
)
{
    int retval = predict_X_old_most_popular(
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        get_ptr(biasA), REAL(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(m), Rf_asInteger(n)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_old_content_based
(
    SEXP predicted,
    SEXP m_new, SEXP k,
    SEXP row, /* <- optional */
    SEXP col,
    SEXP m_orig, SEXP n_orig,
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP C, SEXP C_bias,
    SEXP Bm, SEXP biasB,
    SEXP glob_mean,
    SEXP nthreads
)
{
    int retval = predict_X_old_content_based(
        REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(m_new), Rf_asInteger(k),
        get_ptr_int(row), /* <- optional */
        get_ptr_int(col),
        Rf_asInteger(m_orig), Rf_asInteger(n_orig),
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        REAL(C), get_ptr(C_bias),
        get_ptr(Bm), get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_old_offsets_explicit
(
    SEXP row, SEXP col, SEXP predicted,
    SEXP Am, SEXP biasA,
    SEXP Bm, SEXP biasB,
    SEXP glob_mean,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP m, SEXP n,
    SEXP nthreads
)
{
    int retval = predict_X_old_offsets_explicit(
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        REAL(Am), get_ptr(biasA),
        REAL(Bm), get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        Rf_asInteger(m), Rf_asInteger(n),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_old_offsets_implicit
(
    SEXP row, SEXP col, SEXP predicted,
    SEXP Am,
    SEXP Bm,
    SEXP k,
    SEXP m, SEXP n,
    SEXP nthreads
)
{
    int retval = predict_X_old_offsets_implicit(
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        REAL(Am),
        REAL(Bm),
        Rf_asInteger(k),
        Rf_asInteger(m), Rf_asInteger(n),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_predict_X_new_collective_explicit
(
    /* inputs for predictions */
    SEXP m_new,
    SEXP row, SEXP col, SEXP predicted,
    SEXP nthreads,
    /* inputs for factors */
    SEXP user_bias,
    SEXP U, SEXP m_u, SEXP p,
    SEXP NA_as_zero_U, SEXP NA_as_zero_X,
    SEXP nonneg,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP Ub, SEXP m_ubin, SEXP pbin,
    SEXP C, SEXP Cb,
    SEXP glob_mean, SEXP biasB,
    SEXP U_colmeans,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Xfull, SEXP n,
    SEXP Wfull, SEXP Wsp,
    SEXP B,
    SEXP Bi, SEXP add_implicit_features,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP n_max, SEXP include_all_X,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP TransCtCinvCt,
    SEXP CtCw,
    SEXP B_plus_bias
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        lam_unique = REAL(l1_lam);

    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);


    int retval = predict_X_new_collective_explicit(
        /* inputs for predictions */
        Rf_asInteger(m_new),
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(nthreads),
        /* inputs for factors */
        (bool) Rf_asLogical(user_bias),
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_X),
        (bool) Rf_asLogical(nonneg),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(Ub), Rf_asInteger(m_ubin), Rf_asInteger(pbin),
        get_ptr(C), get_ptr(Cb),
        Rf_asReal(glob_mean), get_ptr(biasB),
        get_ptr(U_colmeans),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int( ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Xfull), Rf_asInteger(n),
        weight,
        REAL(B),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(TransCtCinvCt),
        get_ptr(CtCw),
        get_ptr(B_plus_bias)
    );


    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_new_collective_implicit
(
    /* inputs for predictions */
    SEXP m_new,
    SEXP row, SEXP col, SEXP predicted,
    SEXP nthreads,
    /* inputs for factors */
    SEXP U, SEXP m_u, SEXP p,
    SEXP NA_as_zero_U,
    SEXP nonneg,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP B, SEXP n,
    SEXP C,
    SEXP U_colmeans,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam, SEXP alpha, SEXP w_main, SEXP w_user,
    SEXP w_main_multiplier,
    SEXP apply_log_transf,
    SEXP BeTBe,
    SEXP BtB,
    SEXP BeTBeChol
)
{
    int retval = predict_X_new_collective_implicit(
        /* inputs for predictions */
        Rf_asInteger(m_new),
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(nthreads),
        /* inputs for factors */
        get_ptr(U), Rf_asInteger(m_u), Rf_asInteger(p),
        (bool) Rf_asLogical(NA_as_zero_U),
        (bool) Rf_asLogical(nonneg),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        REAL(B), Rf_asInteger(n),
        get_ptr(C),
        get_ptr(U_colmeans),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asReal(lam), Rf_asReal(l1_lam), Rf_asReal(alpha), Rf_asReal(w_main), Rf_asReal(w_user),
        Rf_asReal(w_main_multiplier),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(BeTBe),
        get_ptr(BtB),
        get_ptr(BeTBeChol)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_new_content_based
(
    SEXP predicted,
    SEXP m_new, SEXP n_new, SEXP k,
    SEXP row, SEXP col, /* <- optional */
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP II, SEXP q,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP I_csr_p, SEXP I_csr_i, SEXP I_csr,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP glob_mean,
    SEXP nthreads
)
{
    int retval = predict_X_new_content_based(
        REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(m_new), Rf_asInteger(n_new), Rf_asInteger(k),
        get_ptr_int(row), get_ptr_int(col), /* <- optional */
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(II), Rf_asInteger(q),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t) Rf_xlength(I_sp),
        get_ptr_size_t(I_csr_p), get_ptr_int(I_csr_i), get_ptr(I_csr),
        REAL(C), get_ptr(C_bias),
        REAL(D), get_ptr(D_bias),
        Rf_asReal(glob_mean),
        Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_new_offsets_explicit
(
    /* inputs for predictions */
    SEXP m_new, SEXP user_bias,
    SEXP row, SEXP col, SEXP predicted,
    SEXP nthreads,
    /* inputs for factors */
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Xfull, SEXP n, /* <- 'n' MUST be passed */
    SEXP Wfull, SEXP Wsp,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP glob_mean, SEXP biasB,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP w_user,
    SEXP lam, SEXP exact,
    SEXP precomputedTransBtBinvBt,
    SEXP precomputedBtB,
    SEXP Bm_plus_bias
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = predict_X_new_offsets_explicit(
        /* inputs for predictions */
        Rf_asInteger(m_new), (bool) Rf_asLogical(user_bias),
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(nthreads),
        /* inputs for factors */
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Xfull), Rf_asInteger(n), /* <- 'n' MUST be passed */
        weight,
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asReal(glob_mean), get_ptr(biasB),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        Rf_asReal(w_user),
        lambda_, lam_unique, (bool) Rf_asLogical(exact),
        get_ptr(precomputedTransBtBinvBt),
        get_ptr(precomputedBtB),
        get_ptr(Bm_plus_bias)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_predict_X_new_offsets_implicit
(
    /* inputs for predictions */
    SEXP m_new,
    SEXP row, SEXP col, SEXP predicted,
    SEXP n_orig,
    SEXP nthreads,
    /* inputs for factors */
    SEXP U, SEXP p,
    SEXP U_row, SEXP U_col, SEXP U_sp,
    SEXP U_csr_p, SEXP U_csr_i, SEXP U_csr,
    SEXP X, SEXP ixA, SEXP ixB,
    SEXP Xcsr_p, SEXP Xcsr_i, SEXP Xcsr,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP k,
    SEXP lam, SEXP alpha,
    SEXP apply_log_transf,
    SEXP precomputedBtB
)
{
    int retval = predict_X_new_offsets_implicit(
        /* inputs for predictions */
        Rf_asInteger(m_new),
        INTEGER(row), INTEGER(col), REAL(predicted), (size_t) Rf_xlength(predicted),
        Rf_asInteger(n_orig),
        Rf_asInteger(nthreads),
        /* inputs for factors */
        get_ptr(U), Rf_asInteger(p),
        get_ptr_int(U_row), get_ptr_int(U_col), get_ptr(U_sp), (size_t) Rf_xlength(U_sp),
        get_ptr_size_t(U_csr_p), get_ptr_int(U_csr_i), get_ptr(U_csr),
        get_ptr(X), get_ptr_int(ixA), get_ptr_int(ixB), (size_t) Rf_xlength(X),
        get_ptr_size_t(Xcsr_p), get_ptr_int(Xcsr_i), get_ptr(Xcsr),
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asInteger(k),
        Rf_asReal(lam), Rf_asReal(alpha),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(precomputedBtB)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_topN_old_collective_explicit
(
    SEXP a_vec, SEXP a_bias,
    SEXP B,
    SEXP biasB,
    SEXP glob_mean,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP n_max, SEXP include_all_X, SEXP nthreads
)
{
    int retval = topN_old_collective_explicit(
        REAL(a_vec), Rf_asReal(a_bias),
        (double*)NULL, (double*)NULL, 0,
        REAL(B),
        get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_old_collective_implicit
(
    SEXP a_vec,
    SEXP B,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP nthreads
)
{
    int retval = topN_old_collective_implicit(
        REAL(a_vec),
        (double*)NULL, 0,
        REAL(B),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_old_most_popular
(
    SEXP user_bias,
    SEXP a_bias,
    SEXP biasB,
    SEXP glob_mean,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n
)
{
    int retval = topN_old_most_popular(
        (bool) Rf_asLogical(user_bias),
        Rf_asReal(a_bias),
        (double*)NULL, 0,
        REAL(biasB),
        Rf_asReal(glob_mean),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_old_content_based
(
    SEXP a_vec, SEXP a_bias,
    SEXP Bm,
    SEXP biasB,
    SEXP glob_mean,
    SEXP k,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP nthreads
)
{
    int retval = topN_old_content_based(
        REAL(a_vec), Rf_asReal(a_bias),
        (double*)NULL, (double*)NULL, 0,
        REAL(Bm),
        get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(k),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_old_offsets_explicit
(
    SEXP a_vec, SEXP a_bias,
    SEXP Bm,
    SEXP biasB,
    SEXP glob_mean,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP nthreads
)
{
    int retval = topN_old_offsets_explicit(
        REAL(a_vec), Rf_asReal(a_bias),
        (double*)NULL, (double*)NULL, 0,
        REAL(Bm),
        get_ptr(biasB),
        Rf_asReal(glob_mean),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_old_offsets_implicit
(
    SEXP a_vec,
    SEXP Bm,
    SEXP k,
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP nthreads
)
{
    int retval = topN_old_offsets_implicit(
        REAL(a_vec),
        (double*)NULL, 0,
        REAL(Bm),
        Rf_asInteger(k),
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

SEXP call_topN_new_collective_explicit
(
    /* inputs for the factors */
    SEXP user_bias,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP u_bin_vec, SEXP pbin,
    SEXP NA_as_zero_U, SEXP NA_as_zero_X,
    SEXP nonneg,
    SEXP C, SEXP Cb,
    SEXP glob_mean, SEXP biasB,
    SEXP U_colmeans,
    SEXP Xa, SEXP ixB,
    SEXP Xa_dense, SEXP n,
    SEXP Wfull, SEXP Wsp,
    SEXP B,
    SEXP Bi, SEXP add_implicit_features,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam,
    SEXP scale_lam, SEXP scale_lam_sideinfo,
    SEXP w_main, SEXP w_user, SEXP w_implicit,
    SEXP n_max, SEXP include_all_X,
    SEXP BtB,
    SEXP TransBtBinvBt,
    SEXP BtXbias,
    SEXP BeTBeChol,
    SEXP BiTBi,
    SEXP CtCw,
    SEXP TransCtCinvCt,
    SEXP B_plus_bias,
    /* inputs for topN */
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);

    double l1_lambda_ = REAL(l1_lam)[0];
    double *l1_lam_unique = NULL;
    if (Rf_xlength(l1_lam) == 6)
        l1_lam_unique = REAL(l1_lam);
    
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = topN_new_collective_explicit(
        /* inputs for the factors */
        (bool) Rf_asLogical(user_bias),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(u_bin_vec), Rf_asInteger(pbin),
        (bool) Rf_asLogical(NA_as_zero_U), (bool) Rf_asLogical(NA_as_zero_X),
        (bool) Rf_asLogical(nonneg),
        get_ptr(C), get_ptr(Cb),
        Rf_asReal(glob_mean), get_ptr(biasB),
        get_ptr(U_colmeans),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Xa_dense), Rf_asInteger(n),
        weight,
        get_ptr(B),
        get_ptr(Bi), (bool) Rf_asLogical(add_implicit_features),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        lambda_, lam_unique,
        l1_lambda_, l1_lam_unique,
        (bool) Rf_asLogical(scale_lam), (bool) Rf_asLogical(scale_lam_sideinfo),
        Rf_asReal(w_main), Rf_asReal(w_user), Rf_asReal(w_implicit),
        Rf_asInteger(n_max), (bool) Rf_asLogical(include_all_X),
        get_ptr(BtB),
        get_ptr(TransBtBinvBt),
        get_ptr(BtXbias),
        get_ptr(BeTBeChol),
        get_ptr(BiTBi),
        get_ptr(CtCw),
        get_ptr(TransCtCinvCt),
        get_ptr(B_plus_bias),
        /* inputs for topN */
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_new_collective_implicit
(
    /* inputs for the factors */
    SEXP n,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP NA_as_zero_U,
    SEXP nonneg,
    SEXP U_colmeans,
    SEXP B, SEXP C,
    SEXP Xa, SEXP ixB,
    SEXP k, SEXP k_user, SEXP k_item, SEXP k_main,
    SEXP lam, SEXP l1_lam, SEXP alpha, SEXP w_main, SEXP w_user,
    SEXP w_main_multiplier,
    SEXP apply_log_transf,
    SEXP BeTBe,
    SEXP BtB,
    SEXP BeTBeChol,
    /* inputs for topN */
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP nthreads
)
{
    int retval = topN_new_collective_implicit(
        /* inputs for the factors */
        Rf_asInteger(n),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        (bool) Rf_asLogical(NA_as_zero_U),
        (bool) Rf_asLogical(nonneg),
        get_ptr(U_colmeans),
        get_ptr(B), get_ptr(C),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        Rf_asInteger(k), Rf_asInteger(k_user), Rf_asInteger(k_item), Rf_asInteger(k_main),
        Rf_asReal(lam), Rf_asReal(l1_lam), Rf_asReal(alpha), Rf_asReal(w_main), Rf_asReal(w_user),
        Rf_asReal(w_main_multiplier),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(BeTBe),
        get_ptr(BtB),
        get_ptr(BeTBeChol),
        /* inputs for topN */
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_new_content_based
(
    /* inputs for the factors */
    SEXP k, SEXP n_new,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP II, SEXP q,
    SEXP I_row, SEXP I_col, SEXP I_sp,
    SEXP I_csr_p, SEXP I_csr_i, SEXP I_csr,
    SEXP C, SEXP C_bias,
    SEXP D, SEXP D_bias,
    SEXP glob_mean,
    /* inputs for topN */
    SEXP outp_ix, SEXP outp_score,
    SEXP nthreads
)
{
    int retval = topN_new_content_based(
        /* inputs for the factors */
        Rf_asInteger(k), Rf_asInteger(n_new),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(II), Rf_asInteger(q),
        get_ptr_int(I_row), get_ptr_int(I_col), get_ptr(I_sp), (size_t) Rf_xlength(I_sp),
        get_ptr_size_t(I_csr_p), get_ptr_int(I_csr_i), get_ptr(I_csr),
        REAL(C), get_ptr(C_bias),
        REAL(D), get_ptr(D_bias),
        Rf_asReal(glob_mean),
        /* inputs for topN */
        INTEGER(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_new_offsets_explicit
(
    /* inputs for factors */
    SEXP user_bias, SEXP n,
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP Xa, SEXP ixB,
    SEXP Xa_dense,
    SEXP Wfull, SEXP Wsp,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP glob_mean, SEXP biasB,
    SEXP k, SEXP k_sec, SEXP k_main,
    SEXP w_user,
    SEXP lam,
    SEXP exact,
    SEXP precomputedTransBtBinvBt,
    SEXP precomputedBtB,
    SEXP Bm_plus_bias,
    /* inputs for topN */
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP nthreads
)
{
    double lambda_ = REAL(lam)[0];
    double *lam_unique = NULL;
    if (Rf_xlength(lam) == 6)
        lam_unique = REAL(lam);
    double *weight = NULL;
    if (Rf_xlength(Wfull))
        weight = REAL(Wfull);
    else if (Rf_xlength(Wsp))
        weight = REAL(Wsp);

    int retval = topN_new_offsets_explicit(
        /* inputs for factors */
        (bool) Rf_asLogical(user_bias), Rf_asInteger(n),
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Xa_dense),
        weight,
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asReal(glob_mean), get_ptr(biasB),
        Rf_asInteger(k), Rf_asInteger(k_sec), Rf_asInteger(k_main),
        Rf_asReal(w_user),
        lambda_, lam_unique,
        (bool) Rf_asLogical(exact),
        get_ptr(precomputedTransBtBinvBt),
        get_ptr(precomputedBtB),
        get_ptr(Bm_plus_bias),
        /* inputs for topN */
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

SEXP call_topN_new_offsets_implicit
(
    /* inputs for factors */
    SEXP u_vec, SEXP p,
    SEXP u_vec_sp, SEXP u_vec_ixB,
    SEXP Xa, SEXP ixB,
    SEXP Bm, SEXP C,
    SEXP C_bias,
    SEXP k,
    SEXP lam, SEXP alpha,
    SEXP apply_log_transf,
    SEXP precomputedBtB,
    /* inputs for topN */
    SEXP include_ix,
    SEXP exclude_ix,
    SEXP outp_ix, SEXP outp_score,
    SEXP n, SEXP nthreads
)
{
    int retval = topN_new_offsets_implicit(
        /* inputs for factors */
        get_ptr(u_vec), Rf_asInteger(p),
        get_ptr(u_vec_sp), get_ptr_int(u_vec_ixB), (size_t) Rf_xlength(u_vec_sp),
        get_ptr(Xa), get_ptr_int(ixB), (size_t) Rf_xlength(Xa),
        get_ptr(Bm), get_ptr(C),
        get_ptr(C_bias),
        Rf_asInteger(k),
        Rf_asReal(lam), Rf_asReal(alpha),
        (bool) Rf_asLogical(apply_log_transf),
        get_ptr(precomputedBtB),
        /* inputs for topN */
        get_ptr_int(include_ix), (int) Rf_xlength(include_ix),
        get_ptr_int(exclude_ix), (int) Rf_xlength(exclude_ix),
        get_ptr_int(outp_ix), get_ptr(outp_score),
        (int) Rf_xlength(outp_ix), Rf_asInteger(n), Rf_asInteger(nthreads)
    );

    SEXP out = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(out)[0] = retval;
    UNPROTECT(1);
    return out; 
}

/* ---------------------------------------------------- */

/* Note: argument limit is 65 */
static const R_CallMethodDef callMethods [] = {
    {"deep_copy", (DL_FUNC) &deep_copy, 1},
    {"as_size_t", (DL_FUNC) &as_size_t, 1},
    /* ---------------------------------------------------- */
    {"call_fit_collective_explicit_lbfgs", (DL_FUNC) &call_fit_collective_explicit_lbfgs, 65},
    {"call_fit_collective_explicit_als", (DL_FUNC) &call_fit_collective_explicit_als, 65},
    {"call_fit_collective_implicit_als", (DL_FUNC) &call_fit_collective_implicit_als, 53},
    {"call_fit_most_popular", (DL_FUNC) &call_fit_most_popular, 20},
    {"call_fit_content_based_lbfgs", (DL_FUNC) &call_fit_content_based_lbfgs, 42},
    {"call_fit_offsets_explicit_lbfgs", (DL_FUNC) &call_fit_offsets_explicit_lbfgs, 52},
    {"call_fit_offsets_explicit_als", (DL_FUNC) &call_fit_offsets_explicit_als, 41},
    {"call_fit_offsets_implicit_als", (DL_FUNC) &call_fit_offsets_implicit_als, 31},
    /* ---------------------------------------------------- */
    {"call_precompute_collective_explicit", (DL_FUNC) &call_precompute_collective_explicit, 31},
    {"call_precompute_collective_implicit", (DL_FUNC) &call_precompute_collective_implicit, 17},
    /* ---------------------------------------------------- */
    {"call_factors_collective_explicit_single", (DL_FUNC) &call_factors_collective_explicit_single, 45},
    {"call_factors_collective_implicit_single", (DL_FUNC) &call_factors_collective_implicit_single, 27},
    {"call_factors_content_based_single", (DL_FUNC) &call_factors_content_based_single, 8},
    {"call_factors_offsets_explicit_single", (DL_FUNC) &call_factors_offsets_explicit_single, 26},
    {"call_factors_offsets_implicit_single", (DL_FUNC) &call_factors_offsets_implicit_single, 17},
    /* ---------------------------------------------------- */
    {"call_factors_collective_explicit_multiple", (DL_FUNC) &call_factors_collective_explicit_multiple, 58},
    {"call_factors_collective_implicit_multiple", (DL_FUNC) &call_factors_collective_implicit_multiple, 38},
    {"call_factors_content_based_mutliple", (DL_FUNC) &call_factors_content_based_mutliple, 14},
    {"call_factors_offsets_explicit_multiple", (DL_FUNC) &call_factors_offsets_explicit_multiple, 37},
    {"call_factors_offsets_implicit_multiple", (DL_FUNC) &call_factors_offsets_implicit_multiple, 27},
    /* ---------------------------------------------------- */
    {"call_impute_X_collective_explicit", (DL_FUNC) &call_impute_X_collective_explicit, 48},
    /* ---------------------------------------------------- */
    {"call_predict_X_old_collective_explicit", (DL_FUNC) &call_predict_X_old_collective_explicit, 15},
    {"call_predict_X_old_collective_implicit", (DL_FUNC) &call_predict_X_old_collective_implicit, 12},
    {"call_predict_X_old_most_popular", (DL_FUNC) &call_predict_X_old_most_popular, 8},
    {"call_predict_X_old_content_based", (DL_FUNC) &call_predict_X_old_content_based, 21},
    {"call_predict_X_old_offsets_explicit", (DL_FUNC) &call_predict_X_old_offsets_explicit, 14},
    {"call_predict_X_old_offsets_implicit", (DL_FUNC) &call_predict_X_old_offsets_implicit, 9},
    /* ---------------------------------------------------- */
    {"call_predict_X_new_collective_explicit", (DL_FUNC) &call_predict_X_new_collective_explicit, 60},
    {"call_predict_X_new_collective_implicit", (DL_FUNC) &call_predict_X_new_collective_implicit, 40},
    {"call_predict_X_new_content_based", (DL_FUNC) &call_predict_X_new_content_based, 28},
    {"call_predict_X_new_offsets_explicit", (DL_FUNC) &call_predict_X_new_offsets_explicit, 38},
    {"call_predict_X_new_offsets_implicit", (DL_FUNC) &call_predict_X_new_offsets_implicit, 28},
    /* ---------------------------------------------------- */
    {"call_topN_old_collective_explicit", (DL_FUNC) &call_topN_old_collective_explicit, 17},
    {"call_topN_old_collective_implicit", (DL_FUNC) &call_topN_old_collective_implicit, 12},
    {"call_topN_old_most_popular", (DL_FUNC) &call_topN_old_most_popular, 9},
    {"call_topN_old_content_based", (DL_FUNC) &call_topN_old_content_based, 12},
    {"call_topN_old_offsets_explicit", (DL_FUNC) &call_topN_old_offsets_explicit, 14},
    {"call_topN_old_offsets_implicit", (DL_FUNC) &call_topN_old_offsets_implicit, 9},
    /* ---------------------------------------------------- */
    {"call_topN_new_collective_explicit", (DL_FUNC) &call_topN_new_collective_explicit, 50},
    {"call_topN_new_collective_implicit", (DL_FUNC) &call_topN_new_collective_implicit, 31},
    {"call_topN_new_content_based", (DL_FUNC) &call_topN_new_content_based, 22},
    {"call_topN_new_offsets_explicit", (DL_FUNC) &call_topN_new_offsets_explicit, 30},
    {"call_topN_new_offsets_implicit", (DL_FUNC) &call_topN_new_offsets_implicit, 20},
    /* ---------------------------------------------------- */
    {NULL, NULL, 0}
}; 

void R_init_cmfrec(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}

#endif
