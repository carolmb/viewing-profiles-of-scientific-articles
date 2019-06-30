library("segmented")
c = c(1, 8, 3, 0.5)
xx<-1:100
xb = c(30, 50, 75)

numSegments = length(c)
b = numeric(numSegments)
for (i in 2:numSegments){
  b[i] = (c[i-1]-c[i])*xb[i-1]+b[i-1]
}

yy = numeric(length(xx))
yy[1:(xb[1]-1)] = c[1]*xx[1:(xb[1]-1)]+b[1]
for (i in 2:(numSegments-1)){
  yy[xb[i-1]:(xb[i]-1)] = c[i]*xx[xb[i-1]:(xb[i]-1)]+b[i]
}
yy[xb[numSegments-1]:length(yy)] = c[numSegments]*xx[xb[numSegments-1]:length(xx)]+b[numSegments]

yr<-yy+rnorm(length(yy),0,3)


#simple example: 1 segmented variable, 1 breakpoint: you do not need to specify
# the starting value for psi
#o<-segmented(out.lm,seg.Z=~x)
#plot(o)
#points(xx, yy)
#points(o)

#1 segmented variable, 2 breakpoints: you have to specify starting values (vector) for psi: 
#o<-segmented(out.lm,seg.Z=~x,psi=c(25,55,70), control=seg.control(display=FALSE))
#slope(o)
#plot(o)
#points(xx, yr)
#points(o)

dati<-data.frame(x=xx,y=yr)
out.lm<-lm(y~x,data=dati)
#o<-segmented(out.lm, seg.Z=~x, psi=c(20, 30, 70))
#o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=100, stop.if.error=FALSE,n.boot=0))
o<-segmented(out.lm, seg.Z=~x, psi=NA, control=seg.control(it.max=100, stop.if.error=FALSE,n.boot=0, K=5))

slope(o)
plot(o)
points(xx, yr)
points(o)

