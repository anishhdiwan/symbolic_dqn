import numpy as np
from genepro.node import Node
import torch
import torch.nn as nn
import math
sign = lambda x: math.copysign(1, x)

class Plus(Node, nn.Module):
  def __init__(self):
    super(Plus,self).__init__()
    self.arity = 2
    self.symb = '+'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return c_outs[0] + c_outs[1]
  
  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return c_outs[0] + c_outs[1]

    if len(c_outs) == 2:
      return c_outs[0] + c_outs[1]
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5

class Minus(Node, nn.Module):
  def __init__(self):
    super(Minus,self).__init__()
    self.arity = 2
    self.symb = '-'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return c_outs[0] - c_outs[1]

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return c_outs[0] - c_outs[1]

    if len(c_outs) == 2:
      return c_outs[0] - c_outs[1]
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5


class Times(Node, nn.Module):
  def __init__(self):
    super(Times,self).__init__()
    self.arity = 2
    self.symb = '*'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.multiply(c_outs[0], c_outs[1])
  
  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.multiply(c_outs[0], c_outs[1])

    if len(c_outs) == 2:
      # return torch.multiply(c_outs[0], c_outs[1])
      return c_outs[0] * c_outs[1]
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5


class Div(Node):
  def __init__(self):
    super(Div,self).__init__()
    self.arity = 2
    self.symb = '/'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid dividing by 0
    sign_b = np.sign(c_outs[1])
    sign_b = np.where(sign_b == 0, 1, sign_b) 
    protected_div = sign_b * c_outs[0] / (1e-9 + np.abs(c_outs[1]))
    return protected_div

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    
    # sign_b = torch.sign(c_outs[1])
    # sign_b = torch.where(sign_b == 0, 1, sign_b) 
    # protected_div = sign_b * c_outs[0] / (1e-9 + torch.abs(c_outs[1]))
    # return protected_div

    if len(c_outs) == 2:
      # sign_b = torch.sign(c_outs[1])
      # sign_b = torch.where(sign_b == 0, 1, sign_b) 
      sign_b = sign(c_outs[1])
      protected_div = sign_b * c_outs[0] / (1e-9 + abs(c_outs[1]))
      return protected_div
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5



class Square(Node):
  def __init__(self):
    super(Square,self).__init__()
    self.arity = 1
    self.symb = '**2'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'after')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.square(c_outs[0])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return c_outs[0]**2

    if len(c_outs) == 1:
      # return c_outs[0]**2

      try:
          ans = c_outs[0]**2
      except OverflowError:
          ans = 1e5
      return ans
      
    else:
      return 1e-5

    
class Cube(Node):
  def __init__(self):
    super(Cube,self).__init__()
    self.arity = 1
    self.symb = '**3'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'after')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.multiply(np.square(c_outs[0]), c_outs[0])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return c_outs[0]**2

    if len(c_outs) == 1:
      return c_outs[0]**3
    else:
      return 1e-5


class Sqrt(Node):
  def __init__(self):
    super(Sqrt,self).__init__()
    self.arity = 1
    self.symb = 'sqrt'

  def _get_args_repr(self, args):
    # let's report also protection
    return "sqrt(abs("+args[0]+"))"

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid arg <= 0
    return np.sqrt(np.abs(c_outs[0]))
  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.sqrt(torch.abs(c_outs[0]))

    if len(c_outs) == 1:
      return math.sqrt(abs(c_outs[0]))
    else:
      return 1e-5



class Log(Node):
  def __init__(self):
    super(Log,self).__init__()
    self.arity = 1
    self.symb = 'log'

  def _get_args_repr(self, args):
    # let's report also protection (to some level of detail)
    return "log(abs("+args[0]+"))"

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid arg <= 0
    protected_log = np.log(np.abs(c_outs[0]) + 1e-9)
    return protected_log

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # protected_log = torch.log(torch.abs(c_outs[0]) + 1e-9)
    # return protected_log

    if len(c_outs) == 1:
      protected_log = math.log(abs(c_outs[0]) + 1e-9)
      return protected_log
    else:
      return 1e-5

class Exp(Node):
  def __init__(self):
    super(Exp,self).__init__()
    self.arity = 1
    self.symb = "exp"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.exp(c_outs[0])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # protected_log = torch.log(torch.abs(c_outs[0]) + 1e-9)
    # return protected_log

    if len(c_outs) == 1:
      # return math.exp(c_outs[0])

      try:
          ans = math.exp(c_outs[0])
      except OverflowError:
          ans = 1e5
      return ans

    else:
      return 1e-5


class Sin(Node):
  def __init__(self):
    super(Sin,self).__init__()
    self.arity = 1
    self.symb = "sin"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.sin(c_outs[0])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.sin(c_outs[0])

    if len(c_outs) == 1:
      return math.sin(c_outs[0])
    else:
      return 1e-5


class Cos(Node):
  def __init__(self):
    super(Cos,self).__init__()
    self.arity = 1
    self.symb = "cos"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.cos(c_outs[0])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.cos(c_outs[0])

    if len(c_outs) == 1:
      return math.cos(c_outs[0])
    else:
      return 1e-5


class Max(Node):
  def __init__(self):
    super(Max,self).__init__()
    self.arity = 2
    self.symb = "max"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,"before")

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.where(c_outs[0]>c_outs[1], c_outs[0], c_outs[1])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.max(c_outs[0], c_outs[1])

    if len(c_outs) == 2:
      return max(c_outs[0], c_outs[1])
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5


class Min(Node):
  def __init__(self):
    super(Min,self).__init__()
    self.arity = 2
    self.symb = "min"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,"before")

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.where(c_outs[0]<c_outs[1], c_outs[0], c_outs[1])

  def get_output_pt(self, X):
    c_outs = self._get_child_outputs_pt(X)
    # return torch.min(c_outs[0], c_outs[1])

    if len(c_outs) == 2:
      return min(c_outs[0], c_outs[1])
    elif len(c_outs) == 1:
      return c_outs[0]
    else:
      return 1e-5


class Feature(Node, nn.Module):
  def __init__(self,id):
    super(Feature,self).__init__()
    self.arity = 0
    self.id = id
    self.symb = 'x_'+str(id)

  def _get_args_repr(self, args):
    return self.symb

  def get_output(self, X):
    return X[:,self.id]
  
  def get_output_pt(self, X):
    return X[:,self.id] #X[self.id] #



class Constant(Node, nn.Module):
  def __init__(self, value : float=None):
    super(Constant,self).__init__()
    self.arity = 0
    self.__value = value
    self.symb = str(value) if value is not None else "const?"
    self.pt_value = None
    
  def get_value(self):
    if not self.__value:
      # sample uniformly between -5 and +5
      self.__value = np.random.uniform()*10 - 5 
      self.symb = str(self.__value)
      self.pt_value = torch.tensor([self.__value],requires_grad=True)
      self.symb = str(self.pt_value.item())
    return self.pt_value.item() 

  def update_symbol(self):
    if not self.__value:
      self.__value = np.random.uniform() * 10 - 5
      self.symb = str(self.__value)
      self.pt_value = torch.tensor([self.__value], requires_grad=True)
    self.symb = str(self.pt_value.item())


  def set_value(self, value : float):
    self.__value = value   
    self.symb = str(value)
    self.pt_value = torch.tensor([self.__value],requires_grad=True)

  def _get_args_repr(self, args):
    # make sure it is initialized
    self.update_symbol()
    self.get_value()
    return self.symb

  def get_output(self, X : np.ndarray) -> np.ndarray:
    # make sure it is initialized
    v = self.get_value()
    return np.repeat(v, len(X))

  def get_output_pt(self, X):
    # make sure it is initialized
    v = self.get_value()
    return self.pt_value.repeat(len(X))
  def set_value(self, value : float):
    self.__value = value
    self.symb = str(value)
    self.pt_value = torch.tensor([self.__value],requires_grad=True)