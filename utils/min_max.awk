BEGIN {
	avg=0;
	min=2147483647;
	max=-2147483648
}
{
	avg+=$col * $col;
	if(($col)>max)
	  	max=($col)
	if(($col)<min)
	  	min=($col)
}
END { 
	print "average ", sqrt(avg/NR), " max a value is ", max, " min a value is ", min
}
