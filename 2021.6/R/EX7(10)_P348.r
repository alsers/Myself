## EX7(10)
library(astsa)
P = 1 : 1024; S = P + 1024
mag.P = log10(apply(eqexp[P, ], 2, max) - apply(eqexp[P, ], 2, min))
mag.S = log10(apply(eqexp[S, ], 2, max) - apply(eqexp[S, ], 2, min))
eq.P = mag.P[1 : 8]; eq.S = mag.S[1 : 8]
ex.P = mag.P[9 : 16]; ex.S = mag.S[9 : 16]; 
NZ.P = mag.P[17]; NZ.S = mag.S[17]


# 计算线性判别方程
cov.eq = var(cbind(eq.P, eq.S))
cov.ex = var(cbind(ex.P, ex.S))
cov.pooled = (cov.ex + cov.eq) / 2

means.eq = colMeans(cbind(eq.P, eq.S))
means.ex = colMeans(cbind(ex.P, ex.S))

slopes.eq = solve(cov.pooled, means.eq)
inter.eq = -sum(slopes.eq * means.eq) / 2

slopes.ex = solve(cov.pooled, means.ex)
inter.ex = -sum(slopes.ex * means.ex) / 2

d.slopes = slopes.eq - slopes.ex
d.inter = inter.eq - inter.ex



# 给新观测分类
new.data = cbind(NZ.P, NZ.S)
d = sum(d.slopes * new.data) + d.inter
post.eq = exp(d) / (1 + exp(d))



# Print(disc function,  posteriors) and plot results
cat(d.slopes[1], 'mag.P + ', d.slopes[2], 'mag.S + ', d.inter, '\n')
cat('P(EQ|data) = ', post.eq, 'P(EX|data) = ', 1 - post.eq, '\n')
plot(eq.P, eq.S, xlim=c(0, 1.5), ylim=c(.75, 1.25), xlab='log mag(p)', ylab="log mag(S)", pch=8, cex=1.1, lwd=2, main='Classification Based on the Magnitude Features')

points(ex.P, ex.S, pch=6, cex=1.1, lwd=2)
points(new.data, cex=1.1, lwd=2)

abline(a=-d.inter / d.slopes[2], b=-d.slopes[1]/d.slopes[2])
text(eq.P-.07, eq.S+.005, label=names(eqexp[1:8]), cex=.8)
text(ex.P-.07, ex.S+.003, label=names(eqexp[9:16]), cex=.8)
text(NZ.P-.07, NZ.S+.003, label=names(eqexp[17]), cex=.8)
legend('topright', c('EQ', 'EX', 'NZ'), pch=c(8, 6, 3), pt.lwd=2, cex=1.1)




# Cross-Validation
all.data = rbind(cbind(eq.P, eq.S), cbind(ex.P, ex.S))
post.eq <- rep(NA, 8) -> post.ex
for(j in 1:16){
    if(j <= 8){samp.eq = all.data[-c(j, 9:16), ]
    samp.ex = all.data[9:16, ]}
    if(j > 8){samp.eq = all.data[1:8, ]
    samp.ex = all.data[-c(j, 1:8), ]}

    df.eq = nrow(samp.eq)-1; df.ex = nrow(samp.ex)-1
    mean.eq = colMeans(samp.eq); mean.ex = colMeans(samp.ex)
    cov.pooled = (df.eq*cov.eq + df.ex*cov.ex) / (df.eq + df.ex)
    slopes.eq = solve(cov.pooled, mean.ex)
    d.slopes = slopes.eq -slopes.ex
    d.inter = inter.eq - inter.ex
    d = sum(d.slopes * all.data[j, ]) + d.inter
    if (j <= 8) post.eq[j] = exp(d) / (1 + exp(d))
    if (j > 8) post.ex[j-8] = 1 / (1 + exp(d))
}

Posterior <- cbind(1:8, post.eq, 1:8, post.ex)
colnames(Posterior) <- c("EQ", "P(EQ|data)", "EX", "P(EX|data)")
round(Posterior, 3) # Results from Cross-validation (not shown)




# EXP7_(11)
P = 1 : 1024; S = P + 1024; p.dim = 2; n = 1024
eq = as.ts(eqexp[, 1:8])
ex = as.ts(eqexp[, 9:16])
nz = as.ts(eqexp[, 17])
f.eq <- array(dim=c(8, 2, 2, 512)) -> f.ex
f.NZ = array(dim=c(2, 2, 512))

# 计算 2x2 Hermitian矩阵行列式
det.c <- function(mat){return(Re(mat[1,1]*mat[2,2]-mat[1,2]*mat[2,1]))}
L = c(15, 13, 5)  # for smoothing
for (i in 1:8){  # compute spectral matrices
    f.eq[i,,,] = mvspec(cbind(eq[P, i], eq[S, i]), spans=L, taper=.5) $ fxx
    f.ex[i,,,] = mvspec(cbind(ex[P, i], ex[S, i]), spans=L, taper=.5) $ fxx}
u = mvspec(cbind(nz[P], nz[S]), spans=L, taper=.5)
f.NZ = u$fxx
bndwidth = u$bandwidth * sqrt(12) * 40
fhat.eq = apply(f.eq, 2:4, mean)
fhat.ex = apply(f.ex, 2:4, mean)
# plot the average spectra
par(mfrow=c(2, 2), mar=c(3, 3, 2, 1), mgp=c(1.6, .6, 0))
Fr = 40 * (1:512) / n
plot(Fr, Re(fhat.eq[1, 1, ]), type='l', xlab='Frequency (Hz)', ylab="")
plot(Fr, Re(fhat.eq[2, 2, ]), type='l', xlab='Frequency (Hz)', ylab="")
plot(Fr, Re(fhat.ex[1, 1, ]), type='l', xlab='Frequency (Hz)', ylab="")
plot(Fr, Re(fhat.ex[2, 2, ]), type='l', xlab='Frequency (Hz)', ylab="")
mtext('Average P-spectra', side=3, line=-1.5, adj=.2, outer=TRUE)
mtext('Earthquakes', side=2, line=-1, adj=.8, outer=TRUE)
mtext('Average s-spectra', side=3, line=-1.5, adj=.82, outer=TRUE)
mtext('Earthquakes', side=2, line=-1, adj=.2, outer=TRUE)
par(fig=c(.75, 1, .75, 1), new=TRUE)
ker = kernel('modified.daniell', L)$coef; ker = c(rev(ker), ker[-1])
plot((-33:33)/40, ker, type='l', ylab="", xlab='', cex.axis=.7, yaxp=c(0, .04, 2))
#choose alpha
Balpha = rep(0, 19)
    for (i in 1:19){alf=i/20
    for (k in 1:256){
        Balpha[i] = Balpha[i] + Re(log(det.c(alf*fhat.ex[,,k] + (1-alf)*fhat.eq[,,k])/det.c(fhat.eq[,,k]))-alf*log(det.c(fhat.ex[,,k])/det.c(fhat.eq[,,k])))}}
alf = which.max(Balpha) / 20  # alpha = .4
# 计算信息准则
rep(0, 17) -> KLDiff -> BDiff -> KLeq -> KLex -> Beq -> Bex
for (i in 1:17){
    if (i <= 8) f0 = f.eq[i,,,]
    if (i > 8 & i <=16) f0 = f.ex[i-8,,,]
    if(i == 17) f0= f.NZ
for (k in 1:256){    # only use freqs out to .25}
    tr = Re(sum(diag(solve(fhat.eq[,,k], f0[,,k]))))
    KLeq[i] = KLeq[i] + tr + log(det.c(fhat.eq[,,k])) - log(det.c(f0[,,k]))
    Beq[i] = Beq[i] + Re(log(det.c(alf*f0[,,k] + (1-alf)*fhat.eq[,,k])/det.c(fhat.eq[,,k])) - alf*log(det.c(f0[,,k])/det.c(fhat.eq[,,k])))
tr = Re(sum(diag(solve(fhat.ex[,,k], f0[,,k]))))
KLex[i] = KLex[i] + tr + log(det.c(fhat.ex[,,k])) - log(det.c(f0[,,k]))
Bex[i] = Bex[i] + Re(log(det.c(alf*f0[,,k] + (1-alf)*fhat.ex[,,k])/det.c(fhat.ex[,,k])) - alf*log(det.c(f0[,,k])/det.c(fhat.ex[,,k])))}
KLDiff[i] = (KLeq[i] - KLex[i]) / n
BDiff[i] = (Beq[i] - Bex[i]) / (2 * n) }
x.b = max(KLDiff) + .1; x.a = min(KLDiff) - .1
y.b = max(BDiff) + .01; y.a = min(BDiff) - .01
dev.new()
plot(KLDiff[9:16], BDiff[9:16], type="p", xlim = c(x.a, x.b), ylim = c(y.a, y.b),
            cex=1.1, lwd=2, xlab="Kullback-Leibler Difference", ylab="chernoff Difference", main="Classification Based on Chernoff and K-L Dustabces", pch=6)
points(KLDiff[1:8], BDiff[1:8], pch=8, cex=1.1, lwd=2)
points(KLDiff[17], BDiff[17], pch=3, cex=1.1, lwd=2)
legend('topleft', legend=c('EQ', 'EX', 'NZ'), pch=c(8, 6, 3), pt.lwd=2)
abline(h=0, v=0, lty=2, col='gray')
text(KLDiff[-c(1, 2, 3, 7, 14)]-.075, BDiff[-c(1, 2, 3, 7, 14)], label=names(eqexp[-c(1, 2, 3, 7, 14)]), cex=.7)
text(KLDiff[c(1, 2, 3, 7, 14)]+.075, BDiff[c(1, 2, 3, 7, 14)], label=names(eqexp[c(1, 2, 3, 7, 14)]), cex=.7)




# EXP7(17)
u = factor(bnrf1ebv)
x = model.matrix(~u-1)[, 1:3]
# x = x[1:1000,]
Var = var(x)
xspec = mvspec(x, spans=c(7, 7), plot=FALSE)
fxxr = Re(xspec$fxx)
# 计算Q
ev = eigen(Var)
Q = ev$vectors%*%diag(1/sqrt(ev$values))%*%t(ev$vectors)
# compute spec envelop and scale vectors
num = xspec$n.used
nfreq = length(xspec$freq)
specenv = matrix(0, nfreq, 1)
beta = matrix(0, nfreq, 3)
for (k in 1:nfreq){
    ev = eigen(2*Q%*%fxxr[,,k]%*%Q/num, symmetric=TRUE)
    specenv[k] = ev$values[1]
    b = Q%*%ev$vectors[,1]
    beta[k, ] = b/sqrt(sum(b^2))}
# output and graphics
frequency = xspec$freq
plot(frequency, 100*specenv, type='l', ylab='Spectral Envelope (%)')
# add signficance threshold to plot
m = xspec$kernel$m
etainv = sqrt(sum(xspec[-m,m]^2))
thresh = 100*(2/num)*exp(qnorm(.999)*etainv)
abline(h=thresh, lty=6, col=4)
# details
output = cbind(frequency, specenv, beta)
colnames(output) = c('freq', 'specenv', 'A', 'C', 'G')
round(output, 3)