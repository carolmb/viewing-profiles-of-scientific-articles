library("segmented")

fileName <- "data/samples.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)
data<-vector("list",length(linn)/2)
for (i in seq(1,length(linn),by=2)){
  months<-as.numeric(strsplit(linn[i],'\t')[[1]])
  views<-as.numeric(strsplit(linn[i+1],'\t')[[1]])
  data[[(i+1)/2]]<-list(months=months,views=views)
  
}
close(conn)

fileName <- "data/samples_breakpoints.txt"
conn <- file(fileName,open="w")
close(conn)

for (i in seq(1,length(linn)/2)) {
	xx<-data[[i]]$months
	yy<-data[[i]]$views
	# print(xx)
	# print(yy)
	
	dati<-data.frame(x=xx,y=yy)
	out.lm<-lm(y~x,data=dati)
	# o<-segmented(out.lm, seg.Z=~x, psi=c(20, 30, 70))
	# o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=100,K=2))
	# o<-segmented(out.lm, seg.Z=~x, psi=NA)
	o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(K=5,it.max=100, stop.if.error=FALSE, n.boot=0))
  
	# slope(o)
	#plot(o)
	# plot(xx,yy)
	# points(o)

	# coefficients(o)
	slopes<-slope(o)$x[,1]
	breakpoints<-o$psi[,2]

	# if (slopes[1]>5000){
	#   slopes <- slopes[2:length(slopes)]
	#   breakpoints <- breakpoints[2:length(breakpoints)]
	# }
	# 
	# dslopes <- abs(diff(slopes))
	# 
	# actualSlopes<-array(0, dim=length(slopes))
	# for (i in seq(1, length(dslopes))){
	#   if (dslopes[i]>threshold){
	#     
	#   }
	# }

	write.table(t(slopes), file = "data/samples_breakpoints.txt", append = TRUE, col.names=FALSE, row.names=FALSE)
	write.table(t(breakpoints), file = "data/samples_breakpoints.txt", append = TRUE, col.names=FALSE, row.names=FALSE)
}