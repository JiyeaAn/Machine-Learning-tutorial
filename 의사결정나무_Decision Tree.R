
## 의사결정 나무_유방암 진단

library(mlbench)
data(BreastCancer)
str(BreastCancer)

table(BreastCancer$Class)
mean(BreastCancer$Class == "benign")
mean(BreastCancer$Class == "malignant")

sum(!complete.cases(BreastCancer))

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

#rpart 패키지_rpart()함수 이용, CART기반의 의사결정나무

library(rpart)
bc.dtree <- rpart(formula = Class ~., data=bc.train, method = "class",
                  parms = list(split="information"))

bc.dtree

#의사결정 나무 그래프 형태로

# install.packages("rpart.plot")
library(rpart.plot)

prp(bc.dtree, type=2, extra=104, fallen.leaves = TRUE, roundint=FALSE,
    main="Decision Tree from Wisconsin Breast Cancer Dataset")


# predict() 함수를 이용해서 테스트데이터 혼동행렬
bc.dtree.pred <- predict(bc.dtree, newdata = bc.test, type = "prob") #범주의 분류확률
head(bc.dtree.pred)

bc.dtree.pred <- predict(bc.dtree, newdata = bc.test, type = "class") #각 케이스별 분류확률
head(bc.dtree.pred)

table(bc.test$Class, bc.dtree.pred, dnn=c("Actual", "Predicted"))
mean(bc.test$Class==bc.dtree.pred)

#적정 나무 크기를 결정
bc.dtree$cptable
printcp(bc.dtree)

#에러가 가장 적은 3번의 분할까지로 나무 성장 프로세스 종료
bc.dtree.pruned <- rpart(Class ~., data=bc.train, method = "class", cp=0.020115,
                         parms = list(split="information"))
bc.dtree.pruned <- prune(bc.dtree, cp = 0.020115)
bc.dtree.pruned

bc.dtree.pruned$cptable

#가지치기까지 끝난 새로운 의사결정 나무 rpart.plot
cols <- ifelse(bc.dtree.pruned$frame$yval==1, "green4", "darkred")
prp(bc.dtree.pruned, type=2, extra=104, fallen.leaves = TRUE, roundint = FALSE,
    branch.lty = 3, col=cols, border.col = cols, shadow.col = "gray",
    split.cex = 1.2, split.suffix = "?",
    split.box.col = "lightgray", split.border.col = "darkgray", 
    split.round = 0.5, 
    main = "Pruned Decision Tree from Wisconsin Breast Cancer Dataset")

# 인수 없이 의사결정 나무 그리기기
# install.packages("rattle")
library(rattle)
fancyRpartPlot(bc.dtree.pruned, sub=NULL)


bc.dtree.pred <- predict(bc.dtree.pruned, newdata = bc.test, type = "class")
table(bc.test$Class, bc.dtree.pred, dnn=c("Actual", "Predicted"))
mean(bc.test$Class==bc.dtree.pred)
