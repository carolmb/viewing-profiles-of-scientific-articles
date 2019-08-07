library("segmented")
library(Metrics)

# p-value for breakpoints
# order by mse

fileName <- "../data/plos_one_data_total.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)
data<-vector("list",length(linn)/2)
for (i in seq(1,length(linn),by=2)){
  months<-as.numeric(strsplit(linn[i],',')[[1]])
  views<-as.numeric(strsplit(linn[i+1],',')[[1]])
  data[[(i+1)/2]]<-list(months=months,views=views)
}
close(conn)

normalize <- function(x) {
	temp <- ((x - min(x)) / (max(x) - min(x)))
	return (temp)
}

get_segmented <- function(dati,filename,itmax,k,stopiferror,nboot,yy,i,xx) {
	result <- tryCatch({


	    out.lm<-lm(y~x,data=dati)

	    o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=itmax, display=FALSE,K=k, stop.if.error=stopiferror, n.boot=nboot))
		y.mse <- mse(yy,broken.line(o,link=FALSE)$fit)

		if (y.mse >= 0.0001) {
			# jpeg(paste('ge/',i))
			# plot(o)
			
			# lines(xx,yy,col='green')
			# dev.off()
		} else {
			slopes<-slope(o)$x[,1]
			breakpoints<-o$psi[,2]
			pred<-predict(o)
			write(i, file = filename, append = TRUE)
			write.table(t(slopes), file = filename, append = TRUE, col.names=FALSE, row.names=FALSE)
			write.table(t(breakpoints), file = filename, append = TRUE, col.names=FALSE, row.names=FALSE)
			write.table(t(pred), file = filename, append = TRUE, col.names=FALSE, row.names=FALSE)
			
			# jpeg(paste('ls/',i))
			# plot(o)
			
			# lines(xx,yy,col='green')
			# dev.off()
		}

		return (o)

	}, error = function(e) {
		print(paste("MY_ERROR:  ",e))
		return ('')
	})
	return (result)
}

tests <- function(itmax,k,stopiferror) {
	filename <- paste("../data/plos_one_total_breakpoints_k",toString(k),"it.max",toString(itmax),"stop.if.error",toString(stopiferror),"_original_data.txt",sep="")
	conn <- file(filename,open="w")
	close(conn)
  
	nboot <- 0
	if (stopiferror) {
		nboot <- 5
	}
	vector.mse <- c()
	vector.p.value <- c()
	for (i in seq(1,length(linn)/2)){
		if (i%%1000==0){
		  print(i)
		}

		xx<-normalize(data[[i]]$months)
		yy<-normalize(data[[i]]$views)
		# xx<-data[[i]]$months
		# yy<-data[[i]]$views

		dati<-data.frame(x=xx,y=yy)
		
		# print(yy)
		y<-get_segmented(dati,filename,itmax,k,stopiferror,nboot,yy,i,xx)
	}
}



# tests(100,3,FALSE)
# [1] 16
# [1] 946
# [1] 87
# [1] 9913
# [1] -0.02208008


# tests(100,5,FALSE)
# [1] 5
# [1] 346
# [1] 87
# [1] 9913
# [1] -0.01375302

# tests(100,4,FALSE)
# [1] 4
# [1] 570
# [1] 87
# [1] 9913
# [1] -0.004059659

# tests(500,4,FALSE)
# tests(300,4,FALSE)

tests(100,4,FALSE)
# tests(500,4,TRUE)
# tests(300,4,TRUE)

# total 10000
# [1] 87 <- casos em que supostamente não tem breakpoint
# [1] 9913

# data/breakpoints_k3it.max100stop.if.errorFALSE.txt
# 0.000109 0.000196
# data/breakpoints_k4it.max1000stop.if.errorFALSE.txt
# 0.000144 0.000511
# data/breakpoints_k4it.max100stop.if.errorFALSE.txt
# 0.000092 0.000205
# data/breakpoints_k4it.max100stop.if.errorTRUE.txt
# 0.000054 0.000108
# data/breakpoints_k4it.max300stop.if.errorFALSE.txt
# 0.000110 0.000289
# data/breakpoints_k4it.max300stop.if.errorTRUE.txt
# 0.000052 0.000071
# data/breakpoints_k4it.max500stop.if.errorFALSE.txt
# 0.000110 0.000289
# data/breakpoints_k4it.max500stop.if.errorTRUE.txt
# 0.000053 0.000091
# data/breakpoints_k5it.max100stop.if.errorFALSE.txt
# 0.000075 0.000160
# data/breakpoints_k6it.max100stop.if.errorFALSE.txt
# 0.000076 0.000270


