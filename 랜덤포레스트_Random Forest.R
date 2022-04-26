
## 랜덤포레스트_유방암 진단

library(mlbench)
data(BreastCancer)
str(BreastCancer)

#불필요한 변수 제거 _ID
bc <- BreastCancer[-1]

#세포조직의 특성을 나타내는 아홉 개 예측변수는 팩터에서 수치형 변수로 변환
bc <- cbind(lapply(bc[-10], function(x) as.numeric(as.character(x))), bc[10])
str(bc)

set.seed(567)
train <- sample(nrow(bc), 0.7*nrow(bc))
bc.train <- bc[train,]
bc.test <- bc[-train,]

table(bc.train$Class)
table(bc.test$Class)

#랜덤포레스트 randomforest()
#install.packages("randomForest")
library(randomForest)

#기본적으로 의사결정 나무의 개수는 500개, 
#M개의 변수가 있을때 루트M개의 변수가 각 노드에서 노드분할 조건을 탐색하기 위해 선택됨 

set.seed(123)
bc.forest <- randomForest(formula=Class ~ ., data = bc.train,
                          na.action=na.roughfix, importance=TRUE)

bc.forest

#예측 정확도

bc.forest.pred <- predict(bc.forest, newdata = bc.test, type="prob")
head(bc.forest.pred)
bc.forest.pred <- predict(bc.forest, newdata = bc.test, type="response")
head(bc.forest.pred)
table(bc.test$Class, bc.forest.pred, dnn=c("Actual", "Predicted"))
mean(bc.test$Class==bc.forest.pred, na.rm = TRUE)

#군집도표 그리기
library(cluster)
clusplot(x=na.omit(bc.test[,-10]), clus = na.omit(bc.forest.pred),
         color = TRUE, shade = TRUE, labels = 4, lines = 0,
         main = "Random Forest Classification from Wisconsin Breast Cancer Dataset")









