#匿名函数
'''
不通过def来声明函数名字，而是通过lambda关键字来定义的函数称为匿名函数
lambda适用于多个参数、一个返回值的情况，可以用一个变量来接收，变量是一个函数对象，执行这个函数对象的结果与执行一个普通函数的结果一样。
lambda函数比普通函数更简洁，且没有声明函数名，上面的代码是用一个变量来接收lambda函数返回的函数对象，并不是lambda函数的名字。
'''
def sum_func(a):
    return a + 1
sum_lambda = lambda a: a + 1

print(sum_func(1))
print(sum_lambda(1))