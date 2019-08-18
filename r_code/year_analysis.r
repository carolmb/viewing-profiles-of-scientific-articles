
fileName <- "data/series.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)
data<-vector("list",length(linn)/2)
for (i in seq(1,length(linn),by=2)){
  months<-as.numeric(strsplit(linn[i],',')[[1]])
  views<-as.numeric(strsplit(linn[i+1],',')[[1]])
  data[[(i+1)/2]]<-list(months=months,views=views)
}
close(conn)

deltas = c()
for (i in seq(1,length(data))){
    months = data[[i]]$months
    delta = months[length(months)] - months[1]
    print(delta)
    deltas <- c(deltas,delta)
}

hist(deltas)