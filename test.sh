#!/bin/bash

# ------------------------------------------------------------------------------------- #
# test on 3DMatch series benchmarks
python test.py --benchmark 3DMatch
python test.py --benchmark 3DLoMatch
python test.py --benchmark 3DMatch_planar
python test.py --benchmark 3DLoMatch_planar
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# test on IndoorLRS series benchmarks
python test.py --benchmark IndoorLRS
python test.py --benchmark IndoorLRS_planar
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# test on ScanNetpp series benchmarks
python test.py --benchmark ScanNetpp_test
python test.py --benchmark ScanNetpp_test_planar
# ------------------------------------------------------------------------------------- #