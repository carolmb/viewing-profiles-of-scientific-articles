library("segmented")
library(Metrics)

# p-value for breakpoints
# order by mse

file1 <- "data/samples.txt"
conn <- file(file1,open="r")
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
  print(length(linn)/2)
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
      slopes<-slope(o)$x[,1]
      breakpoints<-o$psi[,2]
      write(format(y.mse, scientific = FALSE), file = fileName, append = TRUE)
      write.table(t(slopes), file = fileName, append = TRUE, col.names=FALSE, row.names=FALSE)
      write.table(t(breakpoints), file = fileName, append = TRUE, col.names=FALSE, row.names=FALSE)
    }, warning = function(w) {
      # print(paste("MY_WARNING:  ",w))
      warnings()
    }, error = function(e) {
      # print(paste("MY_ERROR:  ",e))
      pdf(paste("k",toString(k),"/it.max",toString(itmax),"_stopiferror",toString(stopiferror),"_",toString(i),sep=""))
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
}

tests(100,3,FALSE)
tests(100,5,FALSE)
tests(100,6,FALSE)

tests(100,4,FALSE)
tests(500,4,FALSE)
tests(300,4,FALSE)

tests(100,4,TRUE)
tests(500,4,TRUE)
tests(300,4,TRUE)

# total 10000
# [1] 87 <- casos em que supostamente não tem breakpoint
# [1] 9913
