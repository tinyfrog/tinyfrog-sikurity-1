import numpy as np

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)] 
        '''
        이미 있는 array와 동일한 모양과 데이터 형태를 유지한 상태에서
        각 원소를 0으로 채워 반환 
        '''
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, =  self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        '''
        생략(ellipsis) 기호를 사용
        넘파이 배열이 가리키는 메모리 위치 고정, 그 위치에 원소들을 덮어 씀
        deep copy를 함
        '''
        return dx