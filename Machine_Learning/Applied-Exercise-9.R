set.seed(1)
library(MASS)
attach(Boston)

############
### Data ###
############
head(Boston)
summary(Boston)

##############
### Part a ###
##############
'Part a: Use the poly() function to fit a cubic polynomial regression to
predict nox using dis. Report the regression output, and plot
the resulting data and polynomial fits. '

lm.fit = lm(nox~poly(dis, 3), data=Boston)
summary(lm.fit)
dislim = range(dis)
dis.grid = seq(from=dislim[1], to=dislim[2], by=0.1)
lm.pred = predict(lm.fit, list(dis=dis.grid))
plot(nox~dis, data=Boston, col="darkgrey")
lines(dis.grid, lm.pred, col="red", lwd=2)

##############
### Part b ###
##############

'Part b: Plot the polynomial fits for a range of different polynomial
degrees (say, from 1 to 10), and report the associated residual
sum of squares.'

all.rss = rep(NA, 10)
for (i in 1:10) {
  lm.fit = lm(nox~poly(dis, i), data=Boston)
  all.rss[i] = sum(lm.fit$residuals^2)
}
all.rss
plot(all.rss)

##############
### Part c ###
##############

'Part c: Perform cross-validation or another approach to select the optimal
degree for the polynomial, and explain your results.'

library(boot)
all.deltas = rep(NA, 10)
for (i in 1:10) {
  glm.fit = glm(nox~poly(dis, i), data=Boston)
  all.deltas[i] = cv.glm(Boston, glm.fit, K=10)$delta[2]
}
plot(1:10, all.deltas, xlab="Degree", ylab="CV error", type="l", pch=20, lwd=2)
abline(v = which.min(all.deltas), col = "blue")

##############
### Part d ###
##############
'Use the bs() function to fit a regression spline to predict nox
using dis. Report the output for the fit using four degrees of
freedom. How did you choose the knots? Plot the resulting fit.'

library(splines)
sp.fit = lm(nox~bs(dis, df=4, knots = c(3,6,9)), data=Boston)
summary(sp.fit)
sp.pred = predict(sp.fit, list(dis=dis.grid))
plot(nox~dis, data=Boston, col="darkgrey")
lines(dis.grid, sp.pred, col="blue", lwd=3)

##############
### Part e ###
##############
'Now fit a regression spline for a range of degrees of freedom, and
plot the resulting fits and report the resulting RSS. Describe the
results obtained.'

all.cv = rep(NA, 20)
for (i in 3:20) {
  lm.fit = lm(nox~bs(dis, df=i), data=Boston)
  all.cv[i] = sum(lm.fit$residuals^2)
}
all.cv[-c(1, 2)]
plot(all.cv[-c(1,2)])

##############
### Part f ###
##############
'Perform cross-validation or another approach in order to select
the best degrees of freedom for a regression spline on this data.
Describe your results.'

all.cv = rep(NA, 20)
for (i in 3:20) {
  lm.fit = glm(nox~bs(dis, df=i), data=Boston)
  all.cv[i] = cv.glm(Boston, lm.fit, K=10)$delta[2]
}
plot(3:20, all.cv[-c(1, 2)], lwd=2, type="l", xlab="df", ylab="CV error")
abline(v = which.min(all.cv), col = "red")
