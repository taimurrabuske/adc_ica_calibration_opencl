set datafile separator ","
set grid
set multiplot layout 2,1
#set yrange [-0.01:0.01]
set ylabel "Error"
plot for [i=2:14] 'error.csv' using 1:i with lines title "e_{".(13-(i-2))."}" lw ((13-i-2)/2)
set autoscale y
set ylabel "dBc"
set xlabel "Conversion/calibration cycles"
set key bottom right
plot "enob.csv" u 1:3 with lines title "SNDR",\
     "" u 1:4 w l title "SFDR"
unset multiplot

