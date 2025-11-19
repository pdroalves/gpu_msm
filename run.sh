#!/usr/bin/bash
make -j && ./test_msm && ./benchmark_msm --benchmark_time_unit=ms 
