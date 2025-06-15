#!/bin/bash

# ------------------------------------------------------------------------------------- #
# evaluation on 3DMatch series benchmarks
python eval_3dmatch.py --benchmark 3DMatch
python eval_3dmatch.py --benchmark 3DLoMatch
python eval_3dmatch.py --benchmark 3DMatch_planar
python eval_3dmatch.py --benchmark 3DLoMatch_planar
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# evaluation on IndoorLRSs series benchmarks
python eval_indoorlrs.py --benchmark IndoorLRS
python eval_indoorlrs.py --benchmark IndoorLRS_planar
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# evaluation on ScanNetpp series benchmarks
python eval_scannetpp.py --benchmark ScanNetpp_test
python eval_scannetpp.py --benchmark ScanNetpp_test_planar
# ------------------------------------------------------------------------------------- #
