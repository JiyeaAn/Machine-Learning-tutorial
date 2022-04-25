
# 나이브베이즈: 국회의원 투표 성향 #

#install.packages("mlbench")
library(mlbench)

data("HouseVotes84")
votes <- HouseVotes84
str(votes)

library(ggplot2)
vote1 <- na.omit(votes[,c(1,2)])
vote1$V1 <- factor(vote1$V1, levels=c("n", "y"), labels=c("No", "Yes"))
ggplot(vote1, aes(x=V1, fill=Class))+
  geom_bar(position = "dodge", width = 0.7) +
  labs(title="Pros and Cons for Vote 1", x="Vote 1", y="Number of Congressmen",
       fill = "Party")

#결측값 핸들링
head(votes)
sum(is.na(votes))

naCount <- function(col, cls) {return(sum(is.na(votes[,col]) & votes$Class ==cls))}
naCount(2, "democrat")
naCount(2, "republican")

#찬성 비율계산
yesProb <- function(col, cls){
  sum.y <- sum(votes[,col]=="y" & votes$Class ==cls, na.rm=TRUE)
  sum.n <- sum(votes[,col]=="n" & votes$Class ==cls, na.rm=TRUE)
  return(sum.y / (sum.y+sum.n))
}

yesProb(2, "democrat")
yesProb(2, "republican")

#정당별 NA값 할당
set.seed(123)
for(i in 2:ncol(votes)){
  if(sum(is.na(votes[,i])) > 0){
    d.na <- which(is.na(votes[,i]) & votes$Class=="democrat")
    r.na <- which(is.na(votes[,i]) & votes$Class=="republican")
    votes[d.na, i] <- ifelse(runif(naCount(i, "democrat"))
                             < yesProb(i, "democrat"), "y", "n")
    votes[r.na, i] <- ifelse(runif(naCount(i, "republican"))
                             < yesProb(i, "republican"), "y", "n") 
  }
}

sum(is.na(votes))
head(votes)


#훈련데이터 70%,  테스트데이터 30%
set.seed(123)
train <- sample(nrow(votes), 0.7*nrow(votes))
votes.train <-votes[train,]
votes.test <-votes[-train,]
table(votes.train$Class)
table(votes.test$Class)


#e1071 패키지 _ naiveBayes()
#install.packages("e1071")
library(e1071)
votes.nb <- naiveBayes(Class ~ ., data = votes.train)
votes.nb


## 각 국회의원에 대한 소속 정단을 예측
#소속 정당 예측 _ predict()
votes.nb.pred <- predict(votes.nb, newdata = votes.test[,-1])
head(votes.nb.pred)


#혼동행렬을 만들어서 성능 평가
table(votes.test$Class, votes.nb.pred, dnn=c("Actual", "Predicted"))
mean(votes.nb.pred==votes.test$Class)


##각 국회의원이 민주당과 공화당 소속의 의원일 확률을 추정하고 이를 바탕으로 소속 정당 예측
votes.nb.pred <- predict(votes.nb, newdata = votes.test[,-1], type="raw")
head(votes.nb.pred)

# 범주별 예측 확률이 50%가 넘는 정당을 선택
votes.nb.pred <- factor(votes.nb.pred[,"republican"] > 0.5, levels = c(FALSE, TRUE),
                        labels = c("democrat", "republican"))

head(votes.nb.pred)

table(votes.test$Class, votes.nb.pred, dnn=c("Actual", "Predicted"))
mean(votes.nb.pred==votes.test$Class)





