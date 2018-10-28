# logisitc
gradLogistic文件使用**R**自带的**ISLR**包里的Smarket数据，将R自带的glm函数的输出结果作为基准，与自行编写的梯度上升法，随机梯度上升法，改善后的随机梯度上升法, newtonRaphson以及R自带的optim优化方法进行比较。比较内容包括系数的mse(mean square error),运行时间以及预测准确率。    
\newline

output文件中附加了运行结果。就预测准确率来看，optim('L-BFGS-B')的效果稍好于随机梯度下降，显著好于其他所有方法。optim('L-BFGS-B')运行速度也是能接受的。   
logistic的梯度求解公式推导可见gradient_logistic_step.ipynb文件,或见我整理的[梯度求解推导](http://nbviewer.jupyter.org/github/ChenShicong/logisitc/blob/master/gradient_logisitc_step.ipynb)。   
logistic的newtonRaphson求解公式推导可见newtonRaphson_logistic_step.ipynb文件,或见我整理的[newtonRaphson求解推导](http://nbviewer.jupyter.org/github/ChenShicong/logisitc/blob/master/newtonRaphson_logisitc_step.ipynb). 本推导参考了[牛顿法解机器学习中的Logistic回归](https://blog.csdn.net/baimafujinji/article/details/51179381), [Newton's method of wiki](https://en.wikipedia.org/wiki/Newton%27s_method), [Logistic Regression and Newton’s Method](https://www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf).
> 本文档部分代码参考自《机器学习实战》.


