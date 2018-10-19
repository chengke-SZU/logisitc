# logisitc
gradLogistic文件使用**R**自带的**ISLR**包里的Smarket数据，将R自带的glm函数的输出结果作为基准，与自行编写的梯度上升法，随机梯度上升法，改善后的随机梯度上升法以及R自带的optim优化方法进行比较。比较内容包括系数的mse(mean square error),以及运行时间。    
output文件中附加了运行结果。就预测准确率来看，optim('L-BFGS-B')的效果都是显著好于其他所有方法。optim('L-BFGS-B')运行速度也是相当不错的。   
logistic的梯度求解公式推导后续再补上。   
> 本文档参考自《机器学习实战》.


