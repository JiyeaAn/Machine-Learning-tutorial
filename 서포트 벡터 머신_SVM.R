
## SVM _ 아이리스 데이터

iris.sub <- subset(iris, select=c("Sepal.Length", "Sepal.Width" , "Species"),
                   subset = Species %in% c("setosa", "virginica"))
iris.sub$Species <- factor(iris.sub$Species)

head(iris.sub)

# 2차원 평면상의 산점도
library(ggplot2)
ggplot(iris.sub, aes(x=Sepal.Length, y=Sepal.Width))+
  geom_point(aes(color=Species, shape=Species), size=2)

# 선형적으로 분리 가능한 것을 확인하였고, SVM을 활용해서 분할
library(e1071)
set.seed(123)
iris.svm <- svm(Species~ ., data=iris.sub, kernel="linear", cost=1, scale =FALSE)
summary(iris.svm)

# 관측값은 svm 객체의 index원소애 인덱스 형태로 저장되어 있음
iris.svm$index
iris.sub[iris.svm$index,]

#서포트벡터의 인덱스를 이용하여 산점도로 표시 (원형이 서포트벡터값)
ggplot(iris.sub, aes(x=Sepal.Length, y=Sepal.Width))+
  geom_point(aes(color=Species, shape=Species), size=2)+
  geom_point(data=iris.sub[iris.svm$index, c(1,2)],
             color = "darkblue", shape = 21, stroke =1.0, size = 5) 

#직선을 그려 초평면으로 나타내기
w <- t(iris.svm$coefs) %*% iris.svm$SV
w
b <- iris.svm$rho
b

ggplot(iris.sub, aes(x=Sepal.Length, y=Sepal.Width))+
  geom_point(aes(color=Species, shape=Species), size=2)+
  geom_point(data=iris.sub[iris.svm$index, c(1,2)],
             color = "darkblue", shape = 21, stroke =1.0, size = 5)+
  geom_abline(intercept = b/w[1,2], slope = -(w[1,1]/w[1,2]),
              color = "dimgray", lty="dashed", lwd =1)


# predict 함수를 이용하여 새로운 데이터에 대한 범주를 예측
iris.svm.pred <- predict(iris.svm, newdata = iris.sub)
head(iris.svm.pred)

table(iris.sub$Species, iris.svm.pred, dnn=c("Actual", "Predicted"))
mean(iris.sub$Species==iris.svm.pred)









