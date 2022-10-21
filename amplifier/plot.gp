set datafile separator ","
set grid

ORDER=6

set multiplot layout 2,1
#set yrange [-0.01:0.01]
set ylabel "b_i, polynomial coefficients"
plot for [i=2:ORDER+1] 'coefficients.csv' using 1:i with lines title "b_{".(ORDER-(i-1))."}" lw ((ORDER+1-i))
set autoscale y
set ylabel "dBc"
set xlabel "Conversion/calibration cycles"
set key bottom right
plot "thd.csv" u 1:2 w l title "THD"
unset multiplot

