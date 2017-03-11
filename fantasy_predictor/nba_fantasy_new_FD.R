data = read.csv("C:\\Users\\ghai7c\\Desktop\\game_logs.csv")
dim(data)
#removing Pos values having #N/A
data$Pos <- as.character(data$Pos)
data <- data[!data$Pos=="#N/A",]
dim(data)
data<-data[complete.cases(data),]
dim(data)

double_triple <- function(n){
  rows = function(tab) lapply(
    seq_len(nrow(tab)),
    function(i) unclass(tab[i,,drop=F])
  )
  
  f <- data.frame(dd = numeric(n), td = numeric(n))
  # for(i in 1:n){
  #   df$x[i] <- i
  #   df$y[i] <- toString(i)
  # }
  # df
  count = 0
  for (i in 1:n) {
    row <- data[i,]
    x <-c(row$TRB,row$PTS,row$AST,row$STL,row$BLK)
    if(length(x[x >=10])==2)
      f$dd[i] <- 1
    else f$dd[i] <- 0
    
    if(length(x[x >=10])==3)
      f$td[i] <- 1
    else f$td[i] <- 0
  }
  return(as.matrix(f))
}


data_modified <-double_triple(nrow(data))
data <- cbind(data, data_modified)
pos <- with(data,data.frame(model.matrix(~Pos,data)))
pos$X.Intercept. <- NULL
data <- cbind(data,pos)
data$Pos <- NULL

data[,c("Rk","DK.Sal","DK.Pts","DK.Val")]<-list(NULL)
data<-data[complete.cases(data),]
dim(data)

data <- data.frame(lapply(data, function(x) as.numeric(as.integer(x))))
str(data)

training_data<-data[1:13000,]
dim(training_data)
test_data<-data[13001:18492,]
dim(test_data)
rownames(test_data) <- 1:nrow(test_data)

coo_mat <- cor(training_data[, sapply(training_data, class) == "numeric"])
attach(training_data)

model_fd <- lm(FD.Pts ~ PosSF+PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+PF+FD.Sal+FD.Val)
summary(model_fd)
vif(model_fd)
anova(model_fd)[["Residuals","Sum Sq"]]
plot(model_fd$fitted.values,model_fd$residuals)
qqnorm(model_fd$residuals)
qqline(model_fd$residuals)
sigma(model_fd)

test_FD.Pts <- as.data.frame(test_data$FD.Pts)
test_pred = predict(model_fd,newdata=test_data,interval="prediction")
test_pred = as.data.frame(test_pred)
mean(test_FD.Pts$`test_data$FD.Pts`)
SSTotal <- sum((test_FD.Pts$`test_data$FD.Pts` - mean(test_FD.Pts$`test_data$FD.Pts`))^2)
SSTotal
SSRes <- sum((test_FD.Pts$`test_data$FD.Pts` - test_pred$fit)^2)
SSRes
SSReg <- sum((test_pred$fit - mean(test_FD.Pts$`test_data$FD.Pts`))^2)
SSReg
n = nrow(test_pred) - 17
n
MSRes <- SSRes/n
MSRes
standard_error <- sqrt(MSRes)
standard_error
pred_RSquared <- 1-(SSRes/SSTotal)
pred_RSquared
pred_RSquared2 <- SSReg/SSTotal
pred_RSquared2

modelg <- gls(DK.Pts ~ PosSF+PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+PF+DK.Sal+DK.Val)
library(nlme)
#outlier analysis

