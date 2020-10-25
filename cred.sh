set +ex

log_file=./profile/log
if [ $# -ge 1 ]
then
	log_file=$1
fi

time ./all examples/credit_risk/config > $log_file
gprof all gmon.out > ./profile/prof_output
rm gmon.out

grep Expected $log_file | awk '{print $3" "$4" "$5}'> ./profile/some

cat ./profile/some |awk '{print $1}' > ./profile/Ii
cat ./profile/some |awk '{print $2}' > ./profile/Oe
cat ./profile/some |awk '{print $3}' > ./profile/Oi
cat ./profile/some |awk '{print ($2 - $3)}' > ./profile/diff

cat ./profile/some | awk -f ./utils/min_max.awk -v col=1
cat ./profile/some | awk -f ./utils/min_max.awk -v col=2
cat ./profile/some | awk -f ./utils/min_max.awk -v col=3
cat ./profile/diff | awk -f ./utils/min_max.awk -v col=1

wc ./profile/*i -l

# gnuplot -persist -e "plot './profile/some' using 1:3:2"
 gnuplot -persist -e "set term dumb; plot './profile/some' using 2, './profile/some' using 3"
