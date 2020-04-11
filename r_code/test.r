library("segmented")
library(Metrics)
traceback()

# p-value for breakpoints
# order by mse

fileName <- "../data/plos_one_2019.txt"
# fileName <- "syn_data.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)
data<-vector("list",length(linn)/3)

global_valid_curves <- 0
global_invalid_curves <- 0

data_index <- 1
for (i in seq(1,length(linn),by=3)){
  doi<-linn[i]
  months<-as.numeric(strsplit(linn[i+1],',')[[1]])
  views<-as.numeric(strsplit(linn[i+2],',')[[1]])
  if (data_index == 139 || data_index == 140 || data_index == 141) {
  	print(months)
  	print(length(months))
  	print(views)
  	print(length(views))
  	pdf(paste(data_index,'.pdf'))
	plot(months,views)
	#lines(months,views)
	dev.off()
  }
  data[[data_index]]<-list(doi=doi,months=months,views=views)
  data_index <- data_index + 1
}

normalize <- function(x) {
	temp <- ((x - min(x)) / (max(x) - min(x)))
	return (temp)
}

get_segmented <- function(dati,doi,filename_curves,filename_mse,itmax,k,stopiferror,nboot,yy,i,xx) {
	result <- tryCatch({

	    out.lm<-lm(y~x,data=dati)
	    o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=itmax, display=FALSE,K=k, stop.if.error=stopiferror, n.boot=nboot))
		y.mse <- mse(yy,broken.line(o,link=FALSE)$fit)
		#write(y.mse,file = filename_mse, append=TRUE)
		
		if (y.mse >= 0.0001) {
			global_invalid_curves <<- global_invalid_curves + 1
		# y.mape <- mape(yy,broken.line(o,link=FALSE)$fit)
		# if (y.mape >= 0.02) {
		# 	# jpeg(paste('ge/',i))
		# 	# plot(o)
			
		# 	# lines(xx,yy,col='green')
		# 	# dev.off()
		# 	print('if')
		 } else {
			# print(global_valid_curves)
			
			global_valid_curves <<- global_valid_curves + 1

			slopes<-slope(o)$x[,1]

			breakpoints<-o$psi[,2]
			pred<-predict(o)
			write(y.mse,file=filename_mse,append=TRUE)
			write(doi, file = filename_curves, append = TRUE)
			write.table(t(slopes), file = filename_curves, append = TRUE, col.names=FALSE, row.names=FALSE)
			write.table(t(breakpoints), file = filename_curves, append = TRUE, col.names=FALSE, row.names=FALSE)
			write.table(t(pred), file = filename_curves, append = TRUE, col.names=FALSE, row.names=FALSE)
			
		}

		return (o)

	}, error = function(e) {
		print("-----------------------------------------------")
		print(i)
		#print(paste("dati:", dati))
		print(paste("MY_ERROR:  ",e))
		#print(paste("Line: ", out.lm))
		#traceback()
		#browser()
		print("-----------------------------------------------")
		return ('')
	})
	return (result)
}

tests <- function(itmax,k,stopiferror) {
	# filename <- paste("../data/plos_one_2019_breakpoints_k4_original1_data.txt",sep="")
	filename <- "segmented_curves_5_7_years.txt"
	filename_mse <- "mse_curves_error_5_7_years.txt"
	conn <- file(filename,open="w")
	close(conn)
	conn2 <- file(filename_mse,open="w")
	close(conn2)
  
	nboot <- 0
	if (stopiferror) {
		nboot <- 5
	}
	vector.mse <- c()
	vector.p.value <- c()
	#for (i in seq(1,length(linn)/3)){
	for (i in seq(1,length(linn)/3)) {
		if (i%%1000==0){
		  print(i)
		}
        
		xx<-data[[i]]$months[2:length(data[[i]]$months)]
		yy<-data[[i]]$views[2:length(data[[i]]$views)]
		# if (max(xx)-min(xx) < 5 && max(xx)-min(xx) > 7)
		# 	continue

		xx<-normalize(xx)
		yy<-normalize(yy)
		
		dati<-data.frame(x=xx,y=yy) 

		y<-get_segmented(dati,doi=data[[i]]$doi,filename,filename_mse,itmax,k,stopiferror,nboot,yy,i,xx)
    }
}

tests(100,4,FALSE)
print(global_valid_curves)
print(global_invalid_curves)


# total 162534
# validos 129872
# invalidos 31995
# problemas 667
