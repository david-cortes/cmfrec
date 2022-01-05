cmake_minimum_required (VERSION 3.12.4)
project (cmfrec VERSION 3.3.1)
set(CMAKE_BUILD_TYPE Release)

### Note: this build script allows configuring 4 things manually:
### (a) Numeric precison (single/double)
### (b) Integer size (standard/64-bit)
### (c) Providing a custom BLAS/LAPACK instead of using FindBLAS and FindLAPACK
### (d) Whether to let the LAPACK Library handle NAN propagation (most of  them can)

### Option (a): Numeric precision
# Library will by default compile with double precision (double),
# but can be changed to single-precision (float) if desired.
option(USE_FLOAT "Build for single-precision (float) type" OFF)
if (USE_FLOAT)
    message(STATUS "Will build for single-precision (float) type.")
    set(real_t float)
    add_compile_definitions(USE_FLOAT)
else()
    message(STATUS "Will build for double-precision (double) type.")
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
option(USE_INT64 "Use larger int width" OFF)
if (USE_INT64)
    message(STATUS "Will use int64_t instead of machine's 'int' type.")
    set(int_t int64_t)
    add_compile_definitions(USE_INT64)
else()
    message(STATUS "Will use machine's native 'int' type.")
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
option(NO_NAN_PROPAGATION "In case the LAPACK library doesn't do proper NAN propagation" OFF)
if (NO_NAN_PROPAGATION)
    add_compile_definitions(FORCE_NO_NAN_PROPAGATION)
endif()

### Option (e): Optimizing for a given CPU architecture
include(CheckCSourceCompiles)
option(USE_MARCH_NATIVE "Build with -march=native" OFF)
if (USE_MARCH_NATIVE AND NOT MSVC)
    set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_c_source_compiles(
        "
        int main(int argc, char **argv)
        {
            return 0;
        }
        "
        SUPPORTS_MARCH_NATIVE
    )
    set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
    if (SUPPORTS_MARCH_NATIVE)
        message(STATUS "Adding flag -march=native.")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mcpu=native")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_MCPU_NATIVE
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_MCPU_NATIVE)
            message(STATUS "Adding flag -mcpu=native.")
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mcpu=native")
        else()
            message(WARNING "Flags -march=native and -mcpu=native not supported by the compiler.")
        endif()
    endif()
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
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "/O2"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/O2")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O2
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O2)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
        endif()
    endif()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "/fp:contract"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/fp:contract")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FPCONTRACT
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FPCONTRACT)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /fp:contract")
        endif()
    endif()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "/fp:except-"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/fp:except-")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FPNOEX
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FPNOEX)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /fp:except-")
        endif()
    endif()
else()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "-O3 "))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-O3")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O3
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O3)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
        else()
            set(CMAKE_REQUIRED_FLAGS "-O2")
            check_c_source_compiles(
                "
                int main(int argc, char **argv)
                {
                    return 0;
                }
                "
                SUPPORTS_O2
            )
            set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
            if (SUPPORTS_O2)
                set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
            endif()
        endif()
    endif()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "-fno-trapping-math"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-fno-trapping-math")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FNTM
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FNTM)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fno-trapping-math")
        endif()
    endif()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "-fno-math-errno"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-fno-math-errno")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FNE
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FNE)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fno-math-errno")
        endif()
    endif()
    if (NOT (${CMAKE_C_FLAGS_RELEASE} MATCHES "-ffp-contract=fast"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-ffp-contract=fast")
        check_c_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FFPCONTRACT
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FFPCONTRACT)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast")
        endif()
    endif()
    check_c_source_compiles(
    "
    void fma_extra(double *a, double w, double *b, int n)
    {
        #pragma clang fp reassociate(on)
        for (int ix = 0; ix < n; ix++)
            a[ix] += w * b[ix] * b[ix];
    }
    int main(int argc, char **argv)
    {
        double a[] = {1.,2.,3.};
        double b[] = {4.,5.,6.};
        double w = 2.;
        int n = 3;
        fma_extra(a, w, b, n);

        return 0;
    }
    "
    SUPPORTS_CLANG_FP_REASSOCIATE
    )
    if (SUPPORTS_CLANG_FP_REASSOCIATE)
        add_compile_definitions(CLANG_FP_REASSOCIATE)
    endif()
endif()

# Link-time optimization if supported
# https://stackoverflow.com/questions/31355692/how-do-i-enable-link-time-optimization-lto-with-cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)
if (LTO_SUPPORTED)
    set_property(TARGET cmfrec PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
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
CHECK_LIBRARY_EXISTS("" "openblas_get_num_threads" "" HAS_OPENBLAS)
if (HAS_OPENBLAS)
    message(STATUS "Using OpenBLAS - will replace its SYR function.")
    add_compile_definitions(AVOID_BLAS_SYR)
    add_compile_definitions(HAS_OPENBLAS)
else()
    CHECK_LIBRARY_EXISTS("" "mkl_get_max_threads" "" HAS_MKL)
    if (HAS_MKL)
        add_compile_definitions(HAS_MKL)
    else()
        CHECK_LIBRARY_EXISTS("" "catlas_saxpby" "" HAS_ATLAS)
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
