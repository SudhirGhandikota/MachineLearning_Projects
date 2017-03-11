#install.packages ("car", dep=T)
data = read.csv("C:\\Users\\ghai7c\\Desktop\\game_logs.csv")
dim(data)
#removing Pos values having #N/A
data$Pos <- as.character(data$Pos)
data <- data[!data$Pos=="#N/A",]
dim(data)
data<-data[complete.cases(data),]
dim(data)
library(car)

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

data[,c("Rk","FD.Sal","FD.Pts","FD.Val")]<-list(NULL)
data<-data[complete.cases(data),]
dim(data)
str(data)


data$DK.Sal <- as.character(data$DK.Sal)
str(data)
data$DK.Sal <- as.integer(data$DK.Sal)
str(data)
data <- data.frame(lapply(data, function(x) as.integer(x)))
data<-data[complete.cases(data),]
str(data)
dim(data)

training_data<-data[1:13000,]
dim(training_data)
test_data<-data[13001:18211,]
dim(test_data)
rownames(test_data) <- 1:nrow(test_data)

coo_mat <- cor(training_data[, sapply(training_data, class) == "numeric"])
attach(training_data)

#partial_rplot(DK.Pts,PosSF,c(PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+PF+DK.Sal+DK.Val+dd+td))
model <- lm(DK.Pts ~ PosPF+PosPG+PosSF+PosSG)
summary(model)
anova(model)[["Residuals","Sum Sq"]]
library(car)
vif(model)
#PosPF and PosSF not significant
model1 <- lm(DK.Pts ~ PosPG+PosSG)
summary(model1)
vif(model1)

#adding Type
model2 <- lm(DK.Pts ~ PosPG+PosSG+Type)
summary(model2)
vif(model2)
#adding Result
model3 <- lm(DK.Pts ~ PosPG+PosSG+Type+Result)
summary(model3)
vif(model3)
#adding MP
model4 <- lm(DK.Pts ~ PosPG+PosSG+Type+Result+MP)
summary(model4)
vif(model4)
#adding FG made
model5 <- lm(DK.Pts ~ PosPG+PosSG+Type+Result+MP+FG)
summary(model5)
vif(model5)
#adding 2P
model6 <- lm(DK.Pts ~ PosPG+PosSG+Type+Result+MP+FG+X2P)
summary(model6)
vif(model6)
#REMOVING FG because of multicollinearity adding 3P
model7 <- lm(DK.Pts ~ PosPG+PosSG+Type+Result+MP+X2P+X3P)
summary(model7)
vif(model7)
#adding FT and removing PosPG
model8 <- lm(DK.Pts ~ PosSG+Type+Result+MP+X2P+X3P+FT)
summary(model8)
vif(model8)
#adding TRB
model9 <- lm(DK.Pts ~ PosSG+Type+Result+MP+X2P+X3P+FT+TRB)
summary(model9)
vif(model9)
#adding AST and removing Result
model10 <- lm(DK.Pts ~ PosSG+Type+MP+X2P+X3P+FT+TRB+AST)
summary(model10)
vif(model10)
#removing Type insignificant and adding STL
model11 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL)
summary(model11)
vif(model11)
#adding BLK
model12 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK)
summary(model12)
vif(model12)
#adding TOV
model13 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV)
summary(model13)
vif(model13)
#adding PF
model14 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF)
summary(model14)
vif(model14)
#adding PTs
model15 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+PTS)
summary(model15)
vif(model15)
#removing PTS due to multicollinearity and adding GmSc
model16 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+GmSc)
summary(model16)
vif(model16)
#removing GmSc due to multicollinearity and adding DK Sal
model17 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal)
summary(model17)
vif(model17)
#adding Dk Val
model18 <- lm(DK.Pts ~ PosSG+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal+DK.Val)
summary(model18)
vif(model18)

anova(model18)[["Residuals","Sum Sq"]]
plot(model18$fitted.values,model18$residuals)
qqnorm(model18$residuals)
qqline(model18$residuals)
#adding dd
model19 <- lm(DK.Pts ~ MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal+DK.Val+dd)
summary(model19)
vif(model19)
anova(model19)[["Residuals","Sum Sq"]]
plot(model19$fitted.values,model19$residuals)
qqnorm(model19$residuals)
#adding td
model20 <- lm(DK.Pts ~ PosSF+PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal+DK.Val+dd+td)
summary(model20)
vif(model20)
anova(model20)[["Residuals","Sum Sq"]]
plot(model20$fitted.values,model20$residuals)
qqnorm(model20$residuals)
qqline(model20$residuals)

#removing TOV insignificant
model20 <- lm(DK.Pts ~ PosSF+PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal+DK.Val+dd+td)
summary(model20)
vif(model20)
anova(model20)[["Residuals","Sum Sq"]]
plot(model20, which=1)
plot(model20, which=2)
plot(model20, which=6)
sigma(model20)
#standardizing DK.Sal
DK_sal_mean = mean(data$DK.Sal)
DK_sal_sd = sd(data$DK.Sal)
data['DK_sal_z'] = (data$DK.Sal - DK_sal_mean)/DK_sal_sd
#Standardizing DK.Val
DK_val_mean = mean(data$DK.Val)
DK_val_sd = sd(data$DK.Val)
data['DK_val_z'] = (data$DK.Val - DK_sal_mean)/DK_sal_sd

attach(data)

#trying standardized model
model21 <- lm(DK.Pts ~ PosSF+PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK_sal_z+DK_val_z+dd+td)
summary(model21)
vif(model21)
anova(model21)[["Residuals","Sum Sq"]]
plot(model21, which=1)
plot(model21, which=2)
plot(model21, which=6)
sigma(model21)#no change in standard error

test_DF.Pts <- as.data.frame(test_data$DK.Pts)
test_pred = predict(model20,newdata=test_data,interval="prediction")
test_pred = as.data.frame(test_pred)
mean(test_DF.Pts$`test_data$DK.Pts`)
SSTotal <- sum((test_DF.Pts$`test_data$DK.Pts` - mean(test_DF.Pts$`test_data$DK.Pts`))^2)
SSTotal
SSRes <- sum((test_DF.Pts$`test_data$DK.Pts` - test_pred$fit)^2)
SSRes
SSReg <- sum((test_pred$fit - mean(test_DF.Pts$`test_data$DK.Pts`))^2)
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

cooks_distance = cooks.distance(model20)
cooks_distance

#outliers <- outlierTest(model20)
#str(outliers)
#outliers

plot(cooks_distance, pch="*", cex=2, main="Influential Obs by Cooks distance") 
abline(h = 5*mean(cooks_distance, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooks_distance)+1, y=cooks_distance, labels=ifelse(cooks_distance>4*mean(cooks_distance, na.rm=T),names(cooks_distance),""), col="red")



#removing outliers
new_data <- data[-c(379,10726,13731,10344,6691,6391,17484,1964,13153,829,7579,9665,7571),]
dim(new_data)
model21 <- lm(DK.Pts ~ PosPF+PosPG+PosSG+Type+MP+X2P+X3P+FT+TRB+STL+BLK+TOV+PF+DK.Sal+dd+td, data=new_data)
summary(model21)
vif(model21)
anova(model21)[["Residuals","Sum Sq"]]
plot(model21, which=1)
plot(model21, which=2)
plot(model21, which=6)
sigma(model21)
#no changes in model. Infact model became worse
