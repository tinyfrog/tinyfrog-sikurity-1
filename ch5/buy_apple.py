from ch5.layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

# Layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forwarding
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price) # 220

# backwarding
dprice = 1 # dL / dz
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax) # 2.2 110 200

