library("rjson")
library(Metrics)
library("segmented")

# traceback()

# p-value for breakpoints
# order by mse

fileName <- "../others/papers_plos_data_time_series2_filtered_html.json"
input_content <- fromJSON(file = fileName)

# fileName <- "../data/plos_one_2019.txt"
# fileName <- "../k3/average_curves_label_k3_xs_ys.txt"
# fileName <- "syn_data.txt"
# conn <- file(fileName,open="r")
# linn <-readLines(conn)
# data<-vector("list",length(linn)/3)

global_valid_curves <- 0
global_invalid_curves <- 0

normalize <- function(x) {
	temp <- ((x - min(x)) / (max(x) - min(x)))
	return (temp)
}

get_segmented <- function(dati,doi,filename_curves,filename_mse,itmax,k,stopiferror,nboot,yy,i,xx) {
	result <- tryCatch({

	    out.lm<-lm(y~x,data=dati)
	    o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=itmax, display=FALSE,K=k, stop.if.error=stopiferror, n.boot=nboot))
		y.mse <- mse(yy,broken.line(o,link=FALSE)$fit)
		
		if (y.mse >= 0.0001) {
		 	global_invalid_curves <<- global_invalid_curves + 1
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
        print(xx)
        print(yy)
		# print(paste("Line: ", out.lm))
		#traceback()
		#browser()
		print("-----------------------------------------------")
		return ('')
	})
	return (result)
}

tests <- function(itmax,k,stopiferror) {
	# filename <- paste("../data/plos_one_2019_breakpoints_k4_original1_data.txt",sep="")
	# filename <- "segmented_curves_5_7_years.txt"
	# filename_mse <- "mse_curves_error_5_7_years.txt"
	# filename <- "k3_test.txt"
	# filename_mse <- "k3_msg_test.txt"
	# filename <- "segmented_curves_ori_original.txt"
	# filename_mse <- "mse_curves_ori_original.txt"



	filename <- "segmented_curves_html.txt"
	filename_mse <- "mse_curves_html.txt"
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
	for (i in seq(1,length(input_content))) {
		# if (i%%1000==0){
		#  print(i)
		#}

        if (length(input_content[[i]][[1]]$months) > 1) {

            doi<- input_content[[i]][[2]]$DI

            xx<-input_content[[i]][[1]]$months[2:length(input_content[[i]][[1]]$months)]
            
            #l1<-input_content[[i]][[1]]$html_views[2:length(input_content[[i]][[1]]$html_views)]
            #l2<-input_content[[i]][[1]]$pdf_views[2:length(input_content[[i]][[1]]$pdf_views)]
            #yy<-unlist(lapply(seq_along(l1),function(i) unlist(l1[i])+unlist(l2[i])))
            yy<-input_content[[i]][[1]]$html_views[2:length(input_content[[i]][[1]]$html_views)]
            
            xx<-normalize(xx)
            yy<-normalize(yy)

            dati<-data.frame(x=xx,y=yy)

            y<-get_segmented(dati,doi=doi,filename,filename_mse,itmax,k,stopiferror,nboot,yy,i,xx)
        }
    }
}

# tests(100,2,FALSE)
tests(100,4,FALSE)
print(global_valid_curves)
print(global_invalid_curves)


# total 162534
# validos 129872
# invalidos 31995
# problemas 667
