library("segmented")
library(Metrics)

# p-value for breakpoints
# order by mse

fileName <- "data/samples.txt"
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
  return ((x - min(x)) / (max(x) - min(x)))
}

tests <- function(itmax,k,stopiferror) {
  fileName <- paste("data/breakpoints_k",toString(k),"it.max",toString(itmax),"stop.if.error",toString(stopiferror),".txt",sep="")
  conn <- file(fileName,open="w")
  close(conn)
  
  nboot <- 0
  if (stopiferror) {
    nboot <- 5
  }

  x.true <- 0
  x.false <- 0
  env=new.env()
  assign("x.true", 0, env=env)
  assign("x.false", 0, env=env)
  print(length(linn))
  vector.mse <- c()
  vector.p.value <- c()
  for (i in seq(1,length(linn)/2)){
    if (i%%1000==0){
      print(i)
    }

    xx<-normalize(data[[i]]$months)
    yy<-normalize(data[[i]]$views)

    dati<-data.frame(x=xx,y=yy)
    out.lm<-lm(y~x,data=dati)

    # o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=100, stop.if.error=FALSE,n.boot=0))
    
    p.value <- davies.test(out.lm,~x)$p.value
    if (p.value > 0.05) {
        x.true = x.true + 1
      } else {
        x.false = x.false + 1
      }

    result = tryCatch({
      o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=itmax, display=FALSE,K=k, stop.if.error=stopiferror, n.boot=nboot))
    
      y.mse <- mse(yy,broken.line(o,link=FALSE)$fit)
      if (y.mse > 0.0001) {
        pdf(paste("k",toString(k),"true/g0.0001/",toString(i),sep=""))
      } else {
        pdf(paste("k",toString(k),"true/le0.0001/",toString(i),sep=""))
      }
      plot(xx,yy)
      points(o)
      lines(xx,broken.line(o,link=FALSE)$fit,col="red",type="l",lwd=3)
      dev.off()
      
      slopes<-slope(o)$x[,1]
      breakpoints<-o$psi[,2]
      # print(breakpoints)
      
      p.value <- davies.test(out.lm,~x,values=breakpoints)$p.value
      if (p.value > 0.05) {
        pdf(paste("k",toString(k),"true/accept/",toString(i),sep=""))
        vector.p.value <- c(vector.p.value,0)
      } else {
        pdf(paste("k",toString(k),"true/reject/",toString(i),sep=""))
        vector.p.value <- c(vector.p.value,1)
      }
      vector.mse <- c(vector.mse, y.mse)
      plot(xx,yy)
      points(o)
      lines(xx,broken.line(o,link=FALSE)$fit,col="red",type="l",lwd=3)
      dev.off()
      
      write(format(y.mse, scientific = FALSE), file = fileName, append = TRUE)
      write.table(t(slopes), file = fileName, append = TRUE, col.names=FALSE, row.names=FALSE)
      write.table(t(breakpoints), file = fileName, append = TRUE, col.names=FALSE, row.names=FALSE)
    }, warning = function(w) {
      # print(paste("MY_WARNING:  ",w))
      warnings()
    }, error = function(e) {
      print(paste("MY_ERROR:  ",e))
      
      pdf(paste("k",toString(k),"true/error_it.max",toString(itmax),"_stopiferror",toString(stopiferror),"_",toString(i),sep=""))
      plot(xx,yy)
      write("0\n0\n0",file = fileName,append = TRUE)
      dev.off()

      p.value <- davies.test(out.lm,~x)$p.value
      if (p.value > 0.05) {
          assign("x.true", get("x.true", env=env)+1, env=env)
        } else {
          assign("x.false", get("x.false", env=env)+1, env=env)
        }
    }, finally = {
      
    })
  }
  print(get("x.true", env=env))
  print(get("x.false", env=env))
  print(x.true)
  print(x.false)
  # print(vector.mse)
  # print(vector.p.value)
  cor(vector.mse, y = vector.p.value,
  method = "pearson")
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

tests(100,4,TRUE)
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


