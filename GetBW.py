# -*- coding: utf-8 -*-

import torch as t
from torch.autograd import Variable as v


'''
It's said that input of a net must be a vriable, so m=v(...), but not  so sure

the gradient calculation is output->input,
so input should have requires_grad=True, which stands for input,
btw, variables relevant to input have requires_grad=True automatically

the variable which is set by user(i.e m here) has .grad=None originally
which means: before the first output backward, m.grad=None,
			 but after the first output backward, m.grad has result
'''
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
k = v(t.zeros(1, 2))
k[0, 0] = m[0, 0] ** 2 + 3 * m[0, 1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
j = t.zeros(2, 2)

print('m =\n', m)
print('k =\n', k)
print('j =\n', j)

'''
before the 1st k.backward, m.grad=None, NoneType has no .data
so: no m.grad.data.zero_() before 1st k.backward(...)

{ATTENTION} if u want a 2nd or more backward, 
			MUST have retain_graph=True in the 1st backward
			otherwise, the buffers will be freed and cause errors
'''
k.backward(t.FloatTensor([[1, 0]]), retain_graph=True)	# 1st backward needs r._g.=T. to run again
j[:, 0] = m.grad.data
print('j =\n', j)

'''
after the 1st backward, u need to reset gradients to have correct answers,
so m.grad.data.zero_() is REQUIRED here,
otherwise, the result will be the sum of the former ones, try #m.grad.data.zero_() and u will see.
'''
m.grad.data.zero_()
k.backward(t.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('j =\n', j)

'''
1. by manual calculation:
   given m=(m1, m2), k=(k1,k2), 
   given k1 = m1**2 + 3*m2, k2 = m2**2 + 2*m1
   gradients matrix(also called Jacobian matirx) shuould be:
   -----------------
   [[dk1/dm1, dk2/dm1] |             [[4, 2]
    [dk1/dm2, dk2/dm2] |m1=2,m2=3 ==  [3, 6]]

2. by automatically calculation:
   each backward(with argument vector) returns a gradients vector(ATTENTION: not a matrix)
   relationships:
   {1} argument vector Va:
   	   (i.e k.backward(argument vector))
   	   len(Va) = len(output) Va means weights of each k:
   	   examples:
   	   m=(m1, m2), k=(k1, k2), Va=(Va1, Va2)
	   k.backward(Va) = Va1*[dk1/dm1] + Va2*[dk2/dm1]	(due to the developer, the result will be transposed)
	   						[dk1/dm2]       [dk2/dm2]
'''