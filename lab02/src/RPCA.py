import numpy as np

class MatrixshapeException(Exception):
    pass

class RPCA:
    
    def __init__(self, D, tol=1e-3, out_max_iter=100, in_max_iter=2000):
        self.out_max_iter = out_max_iter
        self.in_max_iter = in_max_iter
        self.D = np.array(D)
        if len(self.D.shape) != 2:
            raise MatrixshapeException("D should be a 2-dim matrix")
        self.A = np.zeros(self.D.shape)
        self.E = np.zeros(self.D.shape)
        self.lamda = 1 / np.sqrt(max(self.D.shape))
        self.Rho = 1.5
        self.mu = 1.25 * np.linalg.norm(self.D)
        self.Y = np.sign(self.D) / self.__J(np.sign(self.D))
        self.tol = tol
    
    def ADMM(self):
        times = 0
        while times <= self.out_max_iter:
            times_2 = 0
            while times_2 <= self.in_max_iter:
                pre_E = self.E
                pre_A = self.A
                U, S, V = np.linalg.svd(self.D + self.Y/self.mu - self.E, full_matrices=False)
                self.A = U @ self.__soft_threshold(1/self.mu, np.diag(S)) @ V
                self.E = self.__soft_threshold(self.lamda/self.mu, 
                                                   self.D-self.A+self.Y/self.mu) 
                times_2 += 1
                if np.linalg.norm(self.A - pre_A) <= self.tol \
                    and np.linalg.norm(self.E - pre_E) <= self.tol:
                    break
                
            self.Y = self.Y + self.mu * (self.D - self.A -self.E)
            self.mu *= self.Rho
            if np.linalg.norm(self.D-self.A-self.E) < 1e-5:
                print(np.linalg.norm(self.D-self.A-self.E))
                break
            times += 1
        return [self.A, self.E]
    
    def __J(self, D):
        return max(np.linalg.norm(D, ord = 2), 
                   np.linalg.norm(D, ord=np.inf)/self.lamda)
    
    def __soft_threshold(self, epsilon, x):
        return np.sign(x) * np.maximum(0, np.abs(x) - epsilon)


    
