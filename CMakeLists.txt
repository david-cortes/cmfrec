cmake_minimum_required (VERSION 3.9)
project (cmfrec VERSION 3.1.2)
set(CMAKE_BUILD_TYPE Release)

### Note: this build script allows configuring 4 things manually:
### (a) Numeric precison (single/double)
### (b) Integer size (standard/64-bit)
### (c) Providing a custom BLAS/LAPACK instead of using FindBLAS and FindLAPACK
### (d) Whether to let the LAPACK Library handle NAN propagation (most of  them can)

### Option (a): Numeric precision
# Library will by default compile with double precision (double),
# but can be changed to single-precision (float) if desired.
set(USE_FLOAT OFF)
if (USE_FLOAT)
    set(real_t float)
    add_compile_definitions(USE_FLOAT)
else()
    set(real_t double) # <- this is the default
    add_compile_definitions(USE_DOUBLE)
endif()

### Option (b): Integer size
# By default, will use 'int' type, but can also use int64_t,
# in which case it will also need to be linked against
# the ILP64 versions of BLAS and LAPACK. Note that the
# 'FindBLAS' script won't search for OpenBLAS'es ILP64,
# so if turning this ON and using something other than MKL,
# must also pass the BLAS and LAPACK link arguments.
set(USE_INT64 OFF)
if (USE_INT64)
    set(int_t int64_t)
    add_compile_definitions(USE_INT64)
else()
    set(int_t int) # <- this is the default
    add_compile_definitions(USE_INT)
endif()

### Option (c): Custom BLAS/LAPACK linkage
# If a custom BLAS/LAPACK is to be provided, please add it here
# by specifing the necessary link arguments. Be aware that it
# should match with the integer size specified above if it was changed.
set(BLAS_LIBRARIES "")
set(LAPACK_LIBRARIES "")

### Option (d): NAN propagation in LAPACK
# This option will avoid passing inputs with any NAN values to the
# LAPACK library, in case the library does not deal with them properly
# such as some earlier versions of OpenBLAS. It is recommended NOT to
# set this on if the LAPACK library follows the rules for NANs
# (that is, the resulting output will not have NANs in any extra place
#  in which it shouldn't according to the inputs that had NAN values).
# Note that it is still assummed that the BLAS library will propagate
# NANs properly according to IEEE754 rules.
set(NO_NAN_PROPAGATION OFF)
if (NO_NAN_PROPAGATION)
    add_compile_definitions(FORCE_NO_NAN_PROPAGATION)
endif()





### ------ Rest of the script is not meant to be user-configurable ----- ###





# Setting C99 standard
set(CMAKE_C_STANDARD 99)

# Setting the list of files to compile
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/cblas_wrappers.c
              ${PROJECT_SOURCE_DIR}/src/collective.c
              ${PROJECT_SOURCE_DIR}/src/common.c
              ${PROJECT_SOURCE_DIR}/src/helpers.c
              ${PROJECT_SOURCE_DIR}/src/lbfgs.c
              ${PROJECT_SOURCE_DIR}/src/offsets.c)
set(BUILD_SHARED_LIBS True)
add_library(cmfrec SHARED ${SRC_FILES})
add_compile_definitions(CMFREC_COMPILE_TIME)

# Adding the internal headers
target_include_directories(cmfrec PRIVATE ${PROJECT_SOURCE_DIR}/src)

# OpenMP for multi-threading
find_package(OpenMP)
if (OpenMP_C_FOUND)
    target_link_libraries(cmfrec PUBLIC OpenMP::OpenMP_C)
else()
    message(STATUS "OpenMP not found - will compile without multi-threading support")
endif()

# Compiler optimizations
if (MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -march=native)
endif()

# Linkage to BLAS and LAPACK
if (BLAS_LIBRARIES STREQUAL "")
    if (USE_INT64)
        set(BLA_VENDOR "Intel10_64ilp")
    endif()
    find_package(BLAS REQUIRED)
endif()
if (LAPACK_LIBRARIES STREQUAL "")
    if (USE_INT64)
        set(BLA_VENDOR "Intel10_64ilp")
    endif()
    find_package(LAPACK REQUIRED)
endif()

# https://stackoverflow.com/questions/59578248/how-do-i-manipulate-cmake-lists-as-sets
list(APPEND union_list ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
list(REMOVE_DUPLICATES union_list)

# See https://github.com/xianyi/OpenBLAS/issues/3237
set(CMAKE_REQUIRED_LINK_OPTIONS  ${union_list})
set(CMAKE_REQUIRED_LIBRARIES  ${union_list})
check_c_source_compiles(
    "
    int openblas_get_num_threads(void);
    int main(int argc, char **argv)
    {
        int nt = openblas_get_num_threads;
        return 0;
    }
    "
    HAS_OPENBLAS
)
if (HAS_OPENBLAS)
    message(STATUS "Using OpenBLAS - will replace its SYR function.")
    add_compile_definitions(AVOID_BLAS_SYR)
    add_compile_definitions(HAS_OPENBLAS)
else()
    check_c_source_compiles(
        "
        int mkl_get_max_threads(void);
        int main(int argc, char **argv)
        {
            int nt = mkl_get_max_threads();
            return 0;
        }
        "
        HAS_MKL
    )
    if (HAS_MKL)
        add_compile_definitions(HAS_MKL)
    else()
        check_c_source_compiles(
            "
            void catlas_saxpby(const int N, const float alpha, const float *X,
                  const int incX, const float beta, float *Y, const int incY);
            int main(int argc, char **argv)
            {
                float dont_optimize_out[] = {1., 2., 3., 4.};
                int nt = catlas_saxpby(4, 1., NULL, 1, 0., dont_optimize_out, 1);
                double s = dont_optimize_out[1] + dont_optimize_out[3];
                printf(\"res:%f\n\", s);
                return 0;
            }
            "
            HAS_ATLAS
        )
        if (HAS_ATLAS)
            message(STATUS "Using ATLAS - will replace its SYR function.")
            message(WARNING "Note: ATLAS multi-threading might make this library very slow.")
            add_compile_definitions(HAS_ATLAS)
            add_compile_definitions(AVOID_BLAS_SYR)
        endif()
    endif()
endif()


target_link_libraries(cmfrec PUBLIC ${union_list})

# Public header with the data types substituted according to what was built
configure_file(${PROJECT_SOURCE_DIR}/include/cmfrec.h.in cmfrec.h @ONLY)
set_target_properties(cmfrec PROPERTIES PUBLIC_HEADER cmfrec.h)

# Install target
include(GNUInstallDirs)

install(TARGETS cmfrec
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_file(cmfrec.pc.in cmfrec.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmfrec.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
