#!/bin/sh
# Build _GymSolver.so for the AlphaZero SAT simulation env.
# Self-contained: pure C++11 + zlib, NO GSL (the Dirichlet noise was reimplemented
# with <random> in core/Const.cc). Run from this directory:
#
#   PYTHON=python3 ./build_so.sh
#
# Override NUMPY_INC if numpy headers are not auto-detected.
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PYINC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
NUMPY_INC="${NUMPY_INC:-$("$PYTHON" -c 'import numpy; print(numpy.get_include())')}"

echo "python headers : $PYINC"
echo "numpy headers  : $NUMPY_INC"

g++ -O2 -fPIC -std=c++11 -w -fpermissive \
    -I. -I"$PYINC" -I"$NUMPY_INC" \
    -shared -o minisat/gym/_GymSolver.so \
    minisat/core/Solver.cc \
    minisat/core/shadow.cc \
    minisat/core/Const.cc \
    minisat/utils/Options.cc \
    minisat/utils/System.cc \
    minisat/simp/SimpSolver.cc \
    minisat/gym/GymSolver.cc \
    minisat/gym/GymSolver_wrap.c++ \
    -lz

echo "built minisat/gym/_GymSolver.so"
