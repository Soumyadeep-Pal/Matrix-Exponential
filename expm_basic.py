import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class _ExpmPadeHelper(object):
    def __init__(self, A_r, A_c):
        self.A_r = A_r
        self.A_c = A_c
        self._A2_r = torch.Tensor()
        self._A2_c = torch.Tensor()
        self._A4_r = torch.Tensor()
        self._A4_c = torch.Tensor()
        self._A6_r = torch.Tensor()
        self._A6_c = torch.Tensor()
   

    @property
    def A2_r(self):
        self._A2_r = torch.mm(self.A_r, self.A_r) - torch.mm(self.A_c, self.A_c)
        return self._A2_r
    
    @property
    def A2_c(self):
        self._A2_c = torch.mm(self.A_r, self.A_c) +   torch.mm(self.A_c, self.A_r)
        return self._A2_c
    

    @property
    def A4_r(self):
        self._A4_r = torch.mm(self.A2_r, self.A2_r) - torch.mm(self.A2_c, self.A2_c)
        return self._A4_r
    
    @property
    def A4_c(self):
        self._A4_c = torch.mm(self.A2_r, self.A2_c) + torch.mm(self.A2_c, self.A2_r)
        return self._A4_c


    @property
    def A6_r(self):
        self._A6_r = torch.mm(self.A2_r, self.A4_r) - torch.mm(self.A2_c, self.A4_c)
        return self._A6_r
    
    @property
    def A6_c(self):
        self._A6_c = torch.mm(self.A2_c, self.A4_r) + torch.mm(self.A2_r, self.A4_c)
        return self._A6_c
    
    




    def pade13_scaled(self, s):
        # s should have a float dtype for calculation of 2**-s
        s = s.float()
        b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.]).to(device)


        B_r = self.A_r * 2**-s
        B2_r = self.A2_r * 2**(-2*s)
        B4_r = self.A4_r * 2**(-4*s)
        
        B6_r = self.A6_r * 2**(-6*s)
        
        B_c = self.A_c * 2**-s
        B2_c = self.A2_c * 2**(-2*s)
        B4_c = self.A4_c * 2**(-4*s)
        B6_c = self.A6_c * 2**(-6*s)
        
        U22_r = b[13]*B6_r + b[11]*B4_r + b[9]*B2_r
        U22_c = b[13]*B6_c + b[11]*B4_c + b[9]*B2_c

        U2_r = torch.mm(B6_r, U22_r) - torch.mm(B6_c, U22_c)
        U2_c = torch.mm(B6_c, U22_r) + torch.mm(B6_r, U22_c)       
        
        
        identity = torch.eye(B_r.shape[0]).to(device)
        
        
        U2_sum_r = U2_r + b[7]*B6_r + b[5]*B4_r +b[3]*B2_r + b[1]*identity
        U2_sum_c = U2_c + b[7]*B6_c + b[5]*B4_c +b[3]*B2_c
        
        U_r = torch.mm(B_r, U2_sum_r) - torch.mm(B_c, U2_sum_c)
        U_c = torch.mm(B_c, U2_sum_r) + torch.mm(B_r, U2_sum_c) 
                
        V22_r = b[12]*B6_r + b[10]*B4_r + b[8]*B2_r
        V22_c = b[12]*B6_c + b[10]*B4_c + b[8]*B2_c
        
        V2_r = torch.mm(B6_r, V22_r) - torch.mm(B6_c, V22_c)
        V2_c = torch.mm(B6_c, V22_r) + torch.mm(B6_r, V22_c)       
    

        V_r = V2_r + b[6]*B6_r + b[4]*B4_r + b[2]*B2_r + b[0]*identity
        V_c = V2_c + b[6]*B6_c + b[4]*B4_c + b[2]*B2_c
        
        return U_r,U_c,V_r, V_c




def _solve_P_Q(U_r,U_c,V_r, V_c, structure=None):

    P_r = U_r + V_r
    Q_r = -U_r + V_r
    
    P_c = U_c + V_c
    Q_c = -U_c + V_c

    Q_up = torch.cat((Q_r, -Q_c), dim = 1)
    Q_down = torch.cat((Q_c, Q_r), dim = 1)
    Q = torch.cat((Q_up, Q_down), dim = 0)
    P = torch.cat((P_r, P_c), dim = 0)


    X, LU = torch.solve(P,Q)
    a = torch.tensor(2, dtype = torch.int)
    mid = len(X)/a
        
    X_r = X[0:mid]
    X_c = X[mid:len(X)]

    return X_r, X_c

def MatrixExp(A_r, A_c):

  s_np = max(int(np.ceil(np.log2(2 / 4.25))), 0)
  s =  torch.tensor([s_np]).to(device) 

  h = _ExpmPadeHelper(A_r, A_c)
  U_r,U_c,V_r, V_c = h.pade13_scaled(s)
 
  X_r, X_c = _solve_P_Q(U_r,U_c,V_r, V_c)

  
  
  for i in range(s):
      real = torch.mm(X_r, X_r) - torch.mm(X_c, X_c)
      comp = torch.mm(X_r, X_c) + torch.mm(X_c, X_r)
      X_r = real
      X_c = comp

  return X_r, X_c
