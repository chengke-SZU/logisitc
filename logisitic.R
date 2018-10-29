
# -*- coding:utf-8 -*- 


# ===== 使用glm函数 ====
library(ISLR)
names(Smarket)
dim(Smarket)
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket ,family=binomial)
summary(glm.fits)

# ==== 准备数据集 ====
X <- Smarket[,2:7]
X <- as.matrix(cbind(1, X)) # 添加截距项,且转化为矩阵
temp <- Smarket[,ncol(Smarket)]
y <- ifelse(temp == "Up", 1, 0) # 将因变量转化为0,1变量



# ==== 利用梯度上升法求解beta ====
gradAscent = function(x, y, alpha = 0.0007, maxCycles = 5000){
  # alpha: 步长
  # maxCycle : 最大迭代次数
  # weights: 模型系数值
  
  weights = rep(1/ncol(x), ncol(x)) # 初始化模型系数值，即权重
  iterations = 0 # 记录迭代次数
  for (i in 1:maxCycles) {
    gradients = t(x) %*% (y - (1/(1+exp(-x%*%weights))))
    weights = weights + alpha * gradients
    iterations = iterations + 1 # 每迭代一次，迭代次数+1
    # 由实验可知，即使设定梯度的平方和<0.000001这么小的数，
    # 迭代次数也仅仅需要74次便达到了.
    # 所以根本不需要跑完maxCycles次便已达到比较理想的结果了
    if (sum((alpha * gradients)^2) < 0.000001)  break
  }
  return(list(weights = weights, iterations = iterations))
}
gradAscent(x = X, y = y)



# === 梯度下降法 ====
gradDescent = function(x, y, alpha = 0.0007, maxCycles = 5000){
  # alpha: 步长
  # maxCycle : 最大迭代次数
  # weights: 模型系数值
  
  weights = rep(1/ncol(x), ncol(x)) # 初始化模型系数值，即权重
  for (i in 1:maxCycles) {
    # 将梯度上升中的梯度改为负梯度
    gradients = -t(x) %*% (y - (1/(1+exp(-x%*%weights))))
    # 将梯度上升中的加法改为减法，从而上升改为下降
    weights = weights - alpha * gradients
  }
  return(weights)
}
gradDescent(x = X, y = y)


# ==== 随机梯度上升 =====
stocGradAscent = function(x, y, alpha = 0.001){
  weights = rep(1/ncol(x), ncol(x))
  # 每次只使用数据的一个样本，并计算其梯度，迭代
  for (i in 1:nrow(x)) {
    h = 1/(1+exp(-x[i, ] %*% weights))
    error = y[i] - h
    gradients = as.numeric(error) * x[i, ]
    weights = weights + alpha * gradients
  }
  return(weights)
}
stocGradAscent(x = X, y = y)
gradDescent(x = X, y = y, maxCycles = 1250)


# ==== 改善后的随机梯度上升 ====
# 由上可知，随机梯度仅仅遍历了数据集的行数(1250)次，
# 而梯度上升法通过修改迭代次数变为1250次时，效果依旧碾压随机梯度
# 原因应该是有一些不可分的点，每次遍历到这些点时，会引起剧烈波动

# 以下，希望通过采取增加迭代次数，设置动态alpha, 在每一轮迭代中，
# 每次遍历完一个数据就将其删除 的方法进行补救
# 若不添加if主动打断loop的话，将迭代nrow(x)*numIter次，
# 如本实验中将是1250*200=25000次,数字很巨大

stocGradAscent1 = function(x, y, numIter = 200){
  weights = rep(1/ncol(x), ncol(x))
  m = nrow(x)
  iterations = 0 # 记录迭代次数
  
  for (i in 1:numIter) {
    # 设定数据行数的index，在后边随机遍历数据所有行时，
    # 每遍历完一行，便删除掉这一行
    dataIndex = 1:m 
    for (j in 1:m) {
      # 设定动态性变化alpha,此alpha可以缓解不可分点带来的波动性
      # 同时alpha随着i,j的增大永远不会变为0
      alpha = 4/(1+i+j) + 0.001 
      randIndex = sample(dataIndex, 1) # 每次从删剩的dataIndex中选取一行
      # -- 计算梯度 --
      h = 1/(1+exp(-x[randIndex, ] %*% weights)) # 用上边选取出来的行计算sigmoid
      error = y[randIndex] - h
      gradients = as.numeric(error) * x[randIndex, ] 
      # -- 更新权重 --
      weights = weights + alpha * gradients
      dataIndex = dataIndex[-randIndex] # 在选中的一轮迭代中，删除迭代用过的行数
      # 此处添加限制条件，
      # 当阈值为0.0001时要迭代51867次
      # 当阈值为0.00001时要迭代234435次
      # 当阈值为0.000001时要迭代246329次
      iterations = iterations + 1 # 记录迭代次数
      if (sum((alpha * gradients)^2) < 0.00001)  break
    }
  }
  return(list(weights = weights, iterations = iterations))
}
stocGradAscent1(x = X, y = y)



# ===== 基于batch下的随机梯度上升 =======
# 本算法综合了梯度上升和随机梯度上升，
# 每次从样本集中抽取numBatch行，捆绑在一起，
# 从而每次迭代利用x的一个子集，故为batch

stocBatchGradAscent = function(x, y, numIter = 2000, numBatch = 10, threshold = 1e-9){
  weights = rep(1/ncol(x), ncol(x))
  m = nrow(x)
  iterations = 0 # 记录迭代次数
  numIter1 = ceiling(m/10) # batch后迭代一次所有样本需要的次数
  
  for (i in 1:numIter) {
    # 设定数据行数的index，在后边随机遍历数据所有行时，
    # 每遍历完一行，便删除掉这一行
    dataIndex = 1:m 
    for (j in 1:numIter1) {
      # 设定动态性变化alpha,此alpha可以缓解不可分点带来的波动性
      # 同时alpha随着i,j的增大永远不会变为0
      alpha = 4/(1+i+j) + 0.001 
      randIndex = sample(dataIndex, numBatch) # 每次从删剩的dataIndex中选取a行
      # -- 计算梯度 --
      h = 1/(1+exp(-x[randIndex, ] %*% weights)) # 用上边选取出来的行计算sigmoid
      error = y[randIndex] - h
      gradients = t(x[randIndex, ]) %*% error
      # -- 更新权重 --
      weights = weights + alpha * gradients
      dataIndex = dataIndex[-randIndex] # 在选中的一轮迭代中，删除迭代用过的行数
      iterations = iterations + 1 # 记录迭代次数
      if (sum((alpha * gradients)^2) < threshold)  break
    }
  }
  return(list(weights = weights, iterations = iterations))
}
stocBatchGradAscent(x = X, y = y)






# ======== newtonRapshon =========
newtonRapshon = function(x, y, alpha = 0.001, maxCycles = 5000) {
  # alpha: 步长
  # maxCycle : 最大迭代次数
  # weights: 模型系数值
  
  weights = rep(1/ncol(x), ncol(x)) # 初始化模型系数值，即权重
  iterations = 0 # 记录迭代次数
  for (i in 1:maxCycles) {
    
    p = 1/(1+exp(-x%*%weights))
    gradients = t(x) %*% (y - p) # 一阶导
    H = t(x)%*%diag(c(p*(p-1)))%*%x # 二阶导
    
    weights = weights - alpha * solve(H)%*%gradients
    iterations = iterations + 1 
    if ((sum(alpha * solve(H)%*%gradients)^2) < 1e-09)  break
  }
  
  return(list(weights = weights, iterations = iterations))
}
newtonRapshon(x = X, y = y)



# ==== 利用R自带optim函数(L-BFGS-B)求解 ====
optimFun = function(x, y){
  
  # --- 自定义损失函数 ---
  Level2 = function(w){
    predVec = X %*% w # 预测值向量
    predVec = 1/(1+exp(-predVec)) # 预测概率向量
    temp = y * log(predVec) + (1-y) * log(1-predVec)
    return(-2*sum(temp))
  }
  
  w = optim(par = rep(0, ncol(X)), Level2,  method = "L-BFGS-B")$par
  return(w)
}
optimFun(x = X, y = y)



# ==== mse of coefficients ====
mse = function(x){
  # 以R自带glm函数输出系数结果作为基准计算mse
  sum((x - coefGlm)^2)/length(coefGlm)
}


# ==== 预测效果函数 =====
# 建模的最终目的不是为了在training中获得好效果，
# 而是希望建立的模型具有很好的泛化能力，
# 所以最终目的是在训练集中得到的模型可以在测试集表现优异
# 本函数输出预测正确率.
predictFun = function(xtest, ytest, beta){
  predValue = 1/(1+exp(-xtest%*%beta))
  predProb = ifelse(predValue > 0.5, 1, 0)
  rate = mean(predProb == ytest)
  return(rate)
}



# ==== 所有方法效果比较 =====

all_methods_pe = function(x, y){
  
  # ---- 切分数据集 -----
  index = sample(c(T,F), nrow(X), replace = T)
  xtrain = x[index, ]; ytrain = y[index]
  xtest = x[!index, ]; ytest = y[!index]
  
  t.glm = system.time({glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume ,
                                   data=Smarket[index, ],family=binomial)})[3]
  coefGlm = coef(glm.fits) # 存储glm的系数预测值
  glm_mse = sum((coefGlm - coefGlm)^2)/length(coefGlm) # 输出值为0
  # glm的预测亦采用自己编写的predictFun函数，效果与R自带predict函数一致
  glm_rate = predictFun(xtest = xtest, ytest = ytest, beta = coef(glm.fits))
  
  
  t.gradAscent = system.time({gradAscent_weight = gradAscent(xtrain, ytrain)})[3]
  gradAscent_mse = sum((gradAscent_weight[[1]] - coefGlm)^2)/length(coefGlm)
  # gradAscent_mse = mse(gradAscent_weight[[1]])
  gradAscent_rate = predictFun(xtest = xtest, ytest = ytest, beta = gradAscent_weight[[1]])
  
  
  t.stocGradAscent = system.time({stocGradAscent_weight = stocGradAscent(xtrain, ytrain)})[3]
  stocGradAscent_mse = sum((stocGradAscent_weight - coefGlm)^2)/length(coefGlm)
  # stocGradAscent_mse = mse(stocGradAscent_weight)
  stocGradAscent_rate = predictFun(xtest = xtest, ytest = ytest, beta = stocGradAscent_weight)
  
  
  t.stocGradAscent1 = system.time({stocGradAscent1_weight = stocGradAscent1(xtrain, ytrain)})[3]
  stocGradAscent1_mse = sum((stocGradAscent1_weight[[1]] - coefGlm)^2)/length(coefGlm)
  # stocGradAscent1_mse = mse(stocGradAscent1_weight[[1]])
  stocGradAscent1_rate = predictFun(xtest = xtest, ytest = ytest, beta = stocGradAscent1_weight[[1]])
  
  
  t.stocBatchGradAscent = system.time({stocBatchGradAscent_weight = stocBatchGradAscent(xtrain, ytrain)})[3]
  stocBatchGradAscent_mse = sum((stocBatchGradAscent_weight[[1]] - coefGlm)^2)/length(coefGlm)
  stocBatchGradAscent_rate = predictFun(xtest = xtest, ytest = ytest, beta = stocBatchGradAscent_weight[[1]])
  
  
  t.newtonRapshon = system.time({newtonRapshon_weight = newtonRapshon(xtrain, ytrain)})[3]
  newtonRapshon_mse = sum((newtonRapshon_weight[[1]] - coefGlm)^2)/length(coefGlm)
  newtonRapshon_rate = predictFun(xtest = xtest, ytest = ytest, beta = newtonRapshon_weight[[1]])
  
  
  t.optimFun = system.time({optimFun_weight = optimFun(xtrain, ytrain)})[3]
  optimFun_mse = sum((optimFun_weight - coefGlm)^2)/length(coefGlm)
  # optimFun_mse = mse(optimFun_weight)
  optimFun_rate = predictFun(xtest = xtest, ytest = ytest, beta = optimFun_weight)
  
  
  re_time = c(t.glm, t.gradAscent, t.stocGradAscent, t.stocGradAscent1, t.stocBatchGradAscent, t.optimFun, t.newtonRapshon)
  names(re_time) = c("glm", "gradAscent", "stocGradAscent", "stocGradAscent1","stocBatchGradAscent", "optimFun", "newtonRapshon")
  
  re_mse = c(glm_mse, gradAscent_mse, stocGradAscent_mse, stocGradAscent1_mse, stocBatchGradAscent_mse, optimFun_mse, newtonRapshon_mse)
  names(re_mse) = c("glm_mse", "gradAscent", "stocGradAscent", "stocGradAscent1","stocBatchGradAscent", "optimFun", "newtonRapshon")
  
  re_rate = c(glm_rate, gradAscent_rate, stocGradAscent_rate, stocGradAscent1_rate, stocBatchGradAscent_rate, optimFun_rate, newtonRapshon_rate)
  names(re_rate) = c("glm", "gradAscent", "stocGradAscent", "stocGradAscent1","stocBatchGradAscent", "optimFun", "newtonRapshon")
  
  return(list(time = re_time, mse = re_mse, rate = re_rate))
}
all_methods_pe(x = X, y = y)



# ==== 循环B次 =====
# 由于切分数据具有随机性，
# 切分数据的不同可能导致预测结果波动大
# 故循环B次，从而减少随机性

# ==== revise ====
# 将循环转为矩阵化

iterFun = function(x, y, B){
  
  finalReslut = list()
  timeMat = NULL
  mseMat = NULL
  rateMat = NULL
    
  for (i in 1:B) {
    temp = all_methods_pe(x, y)
    timeMat = cbind(timeMat, temp[[1]])
    mseMat = cbind(mseMat, temp[[2]])
    rateMat = cbind(rateMat, temp[[3]])
  }
  
  finalReslut$time = rowMeans(timeMat)
  finalReslut$mse = rowMeans(mseMat)
  finalReslut$rate = rowMeans(rateMat)
  
  return(finalReslut)
}
iterFun(x = X, y = y, B = 20)







