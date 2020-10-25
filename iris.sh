set +ex

##
# 1 hidden layer 5 nodes(sig)
# learning rate 0.6
# input unscaled 
# Def.make test macro -DIRIS_DATA=true
##

log_file=./profile/log
if [ $# -ge 1 ]
then
	log_file=$1
fi

time ./prometheus examples/iris/config > $log_file
gprof ./prometheus gmon.out > ./profile/prof_output
rm gmon.out

grep Expected $log_file  >./profile/some

cat ./profile/some 
