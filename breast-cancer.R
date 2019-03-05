library(readr)
library(ggplot2)
library(corrplot)

data <- read.csv("breast-cancer-data.csv", header=T, stringsAsFactors=F)

data$X = NULL
data = data[,-1]

head(data)
dim(data)

table(data$diagnosis)

# Encoding
diagnosisLabel <- c("M", "B")
diagnosis <- data$diagnosis
ordered <- factor(diagnosis, levels = diagnosisLabel)
diagnosisLabelBin <- as.numeric(ordered) - 1
data$diagnosis <- diagnosisLabelBin

table(data$diagnosis)

summary(data[,-1])

diagnosis = data[,1]
data = data[,-1]

corr = cor(data)
corrplot(corr,type="lower",tl.col=1,tl.cex=0.7)

# PCA
pca = princomp(data, cor=T)
summary(pca)
plot(pca)

gof = (pca$sdev)^2/sum((pca$sdev)^2)
sum(gof[1:6])

newdata = pca$scores[,1:6]
newdata = cbind(diagnosis, newdata)
colnames(newdata) = c("diagnosis","p1","p2","p3","p4","p5","p6")

newdata=as.data.frame(newdata)

newdata$p1 = as.numeric(newdata$p1)
newdata$p2 = as.numeric(newdata$p2)
newdata$p3 = as.numeric(newdata$p3)
newdata$p4 = as.numeric(newdata$p4)
newdata$p5 = as.numeric(newdata$p5)
newdata$p6 = as.numeric(newdata$p6)

head(newdata)
