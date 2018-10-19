
# -*- coding:utf-8 -*- 


# ===== 使用glm函数 ====
library(ISLR)
names(Smarket)
dim(Smarket)
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket ,family=binomial)
summary(glm.fits)
coefGlm = coef(glm.fits);coefGlm


# ==== 利用梯度上升法求解beta ====
X <- cbind(Smarket[,2:7])
X <- as.matrix(cbind(1, X)) # 添加截距项,且转化为矩阵
temp <- Smarket[,ncol(Smarket)]
y <- ifelse(temp == "Up", 1, 0) # 将因变量转化为0,1变量

gradAscent = function(x, y, alpha = 0.0007, maxCycles = 5000){
  # alpha: 步长
  # maxCycle : 最大迭代次数
  # weights: 模型系数值
  
  weights = rep(1/ncol(x), ncol(x)) # 初始化模型系数值，即权重
  for (i in 1:maxCycles) {
    gradients = t(x) %*% (y - (1/(1+exp(-x%*%weights))))
    weights = weights + alpha * gradients
  }
  return(weights)
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

stocGradAscent1 = function(x, y, numIter = 200){
  weights = rep(1/ncol(x), ncol(x))
  m = nrow(x)
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
    }
  }
  return(weights)
}
coefSGA1 = stocGradAscent1(x = X, y = y)



# ==== 利用R自带optim函数(L-BFGS-B)求解 ====
optimFun = function(x, y){
  
  # --- 自定义Level2 ---
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
  # 以R自带glm函数输出结果作为基准
  sum((x - coefGlm)^2)
}


# ==== 所有方法效果比较 =====
all_methods_pe = function(x, y){
  
  t.glm = system.time({glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume ,
                                   data=Smarket ,family=binomial)})[3]
  
  t.gradAscent = system.time({gradAscent_weight = gradAscent(x, y)})[3]
  gradAscent_mse = mse(gradAscent_weight)
  
  t.stocGradAscent = system.time({stocGradAscent_weight = stocGradAscent(x, y)})[3]
  stocGradAscent_mse = mse(stocGradAscent_weight)
  
  t.stocGradAscent1 = system.time({stocGradAscent1_weight = stocGradAscent1(x, y)})[3]
  stocGradAscent1_mse = mse(stocGradAscent1_weight)
  
  t.optimFun = system.time({optimFun_weight = optimFun(x, y)})[3]
  optimFun_mse = mse(optimFun_weight)
  
  re_time = c(t.glm, t.gradAscent, t.stocGradAscent, t.stocGradAscent1, t.optimFun)
  names(re_time) = c("glm", "gradAscent", "stocGradAscent", "stocGradAscent1", "optimFun")
  
  re_mse = c(gradAscent_mse, stocGradAscent_mse, stocGradAscent1_mse, optimFun_mse)
  names(re_mse) = c("gradAscent", "stocGradAscent", "stocGradAscent1", "optimFun")
  
  return(list(time = re_time, mse = re_mse))
}
all_methods_pe(x = X, y = y)










