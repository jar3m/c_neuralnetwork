 cat test/src/air_train | awk '{sum+=$0; sumsq+=$0*$0 } END {print sum/NR; print sqrt(sumsq/NR - (sum/NR)^2)}'
