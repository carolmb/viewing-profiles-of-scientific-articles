#usage of my.plotcorr
#much like the my.plotcorr function, this is modified from the plotcorr documentation
#this function requires the ellipse library, though, once installed you don't need to load it - it is loaded in the function
install.packages(c('ellipse'))
library(ellipse)
library(latex2exp)
source('my.plotcorr.R')

# corr.mtcars = matrix(nrow=5,ncol=5)

# alpha l
# corr.mtcars[1,1] <- -0.6931
# corr.mtcars[1,2] <- 0.0861

# corr.mtcars[2,1] <- 0.3387 
# corr.mtcars[2,2] <- -0.3595
# corr.mtcars[2,3] <- 0.3178

# corr.mtcars[3,2] <- 0.2912
# corr.mtcars[3,3] <- -0.5161
# corr.mtcars[3,4] <- -0.0022

# corr.mtcars[4,3] <- 0.3714 
# corr.mtcars[4,4] <- -0.1745
# corr.mtcars[4,5] <- -0.0645

# corr.mtcars[5,4] <- 0.4731
# corr.mtcars[5,5] <- 0.0625

# rowlabs <- c(TeX('$\\l_1$'),TeX('$\\l_2$'),TeX('$\\l_3$'),TeX('$\\l_4$'),TeX('$\\l_5$'))
# collabs <- c(TeX('$\\alpha_1$'),TeX('$\\alpha_2$'),TeX('$\\alpha_3$'),TeX('$\\alpha_4$'),TeX('$\\alpha_5$'))


# alpha alpha
corr.mtcars = matrix(nrow=4,ncol=4)
corr.mtcars[1,1] <- -0.04
corr.mtcars[2,2] <- -0.34
corr.mtcars[3,3] <- -0.28
corr.mtcars[4,4] <- 0.11

# l l
# corr.mtcars = matrix(nrow=4,ncol=4)
# corr.mtcars[1,1] <- -0.53
# corr.mtcars[2,2] <- -0.5
# corr.mtcars[3,3] <- -0.46
# corr.mtcars[4,4] <- -0.44


# Change the column and row names for clarity
# colnames(corr.mtcars) = c(TeX('\alpha_1'),TeX('\\alpha_2'),TeX('\\alpha_3'),TeX('\\alpha_4'),TeX('\\alpha_5'))
# rownames(corr.mtcars) = c('l1','l2','l3','l4','l5')
 
# Standard plot, all ellipses are grey, nothing is put in the diagonal
my.plotcorr(corr.mtcars)
 
# Here we play around with the colors, colors are selected from a list with colors recycled
# Thus to map correlations to colors we need to make a list of suitable colors
# To start, pick the end (and mid) points of a scale, here a red to white to blue for neg to none to pos correlation
colsc=c(rgb(0, 61, 104, maxColorValue=255), 'white', rgb(241, 54, 23, maxColorValue=255))
 
# Build a ramp function to interpolate along the scale, I've opted for the Lab interpolation rather than the default rgb, check the documentation about the differences
colramp = colorRampPalette(colsc, space='Lab')
 
# I'll show two types of color styles using this color ramp
# the first
# Use the same number of colors along the scale for the number of variables
colors = colramp(length(corr.mtcars[1,]))
 
# then plot an example with only ellipses, without a diagonal and with a main title
# the color selection stuff here multiplies the correlations such that they can index individual colors and create a sufficiently large list
# incase you are confused, r allows vector indexing with non-integers by rounding down, i.e. colors[1.8] == colors[1]
my.plotcorr(corr.mtcars, col=colors[5*corr.mtcars + 6], main='Predictor correlations')
 
# the second form
# we could, alternatively, make a scale with 100 points
colors = colramp(100)

print(corr.mtcars)

my.plotcorr(corr.mtcars, col=colors[((corr.mtcars + 1)/2) * 100], diag='ellipse', lower.panel='ellipse', upper.panel='ellipse',main='')
