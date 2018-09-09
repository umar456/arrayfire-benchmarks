
get_size(str) = substr(str, strstrt(str,"/") +1, 99)
set datafile separator ","
set logscale y 
set xtics rotate -45
set key autotitle columnhead
plot './build/randu.data' using 4:xtic(get_size(strcol(1))) with linespoints
