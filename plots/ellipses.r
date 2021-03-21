#usage of my.plotcorr
#much like the my.plotcorr function, this is modified from the plotcorr documentation
#this function requires the ellipse library, though, once installed you don't need to load it - it is loaded in the function
install.packages(c('ellipse'))
install.packages(c('latex2exp'))
library(ellipse)
library(latex2exp)
source('my.plotcorr.r')

corr.mtcars = matrix(nrow=5,ncol=5)

# alpha l
corr.mtcars[1,1] <- -0.55
corr.mtcars[1,2] <- 0.14

corr.mtcars[2,1] <- 0.28 
corr.mtcars[2,2] <- -0.36
corr.mtcars[2,3] <- 0.29

corr.mtcars[3,2] <- 0.26
corr.mtcars[3,3] <- -0.49
corr.mtcars[3,4] <- -0.02

corr.mtcars[4,3] <- 0.36 
corr.mtcars[4,4] <- -0.16
corr.mtcars[4,5] <- -0.05

corr.mtcars[5,4] <- 0.47
corr.mtcars[5,5] <- 0.06

# Here we play around with the colors, colors are selected from a list with colors recycled
# Thus to map correlations to colors we need to make a list of suitable colors
# To start, pick the end (and mid) points of a scale, here a red to white to blue for neg to none to pos correlation
colsc=c(rgb(0, 61, 104, maxColorValue=255), 'white', rgb(241, 54, 23, maxColorValue=255))
 
# Build a ramp function to interpolate along the scale, I've opted for the Lab interpolation rather than the default rgb, check the documentation about the differences
colramp = colorRampPalette(colsc, space='Lab')
 
# I'll show two types of color styles using this color ramp
# the first
# Use the same number of colors along the scale for the number of variables
# colors = colramp(length(corr.mtcars[1,]))
 
# the second form
# we could, alternatively, make a scale with 100 points
colors = colramp(100)

print(corr.mtcars)

my.plotcorr(corr.mtcars, col=colors[((corr.mtcars + 1)/2) * 100], diag='ellipse', lower.panel='ellipse', upper.panel='ellipse',main='')

