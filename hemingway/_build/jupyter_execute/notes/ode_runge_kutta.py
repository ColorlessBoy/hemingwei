#!/usr/bin/env python
# coding: utf-8

# # ODE-Runge-Kutta

# ## Runge-Kutta-RK3

# In[1]:


import sympy as sp
def removeSub(expr):
    if isinstance(expr, sp.Subs):
        return expr.args[0].replace(expr.args[1][0], expr.args[2][0])
    for arg in expr.args:
        expr = expr.replace(arg, removeSub(arg))
    return expr


# In[2]:


x, dx, y, dy = sp.symbols('x \Delta{x} y \Delta{y}')
f, y_ = sp.symbols('f y', cls=sp.Function)


# In[3]:


dy_dx = f(x, y)
dy_dx2 = removeSub(f(x, y_(x)).diff(x).subs(y_(x).diff(x), dy_dx).subs(y_(x), y))
dy_dx3 = removeSub(f(x, y_(x)).diff(x, 2).subs(y_(x).diff(x, 2), dy_dx2).subs(y_(x).diff(x), dy_dx).subs(y_(x), y)).expand().simplify()
y_taylor = y_(x) + dy_dx*dx + dy_dx2*dx**2/2 + dy_dx3*dx**3/6
display(dy_dx, dy_dx2, dy_dx3, y_taylor)


# In[4]:


pdf1 = f(x, y).diff(x)*dx + f(x, y).diff(y)*dy
pdf2 = (pdf1.diff(x)*dx + pdf1.diff(y)*dy).expand().simplify()
f_taylor = f(x, y) + pdf1 + pdf2/2
display(pdf1, pdf2, f_taylor)


# $$
# \begin{aligned}
# y_{n+1} =& y_n + h(C_1 K_1 + C_2 K_2 + C_3 K_3)\\
# K_1 =& f(x_n, y_n)\\
# K_2 =& f(x_n + a_2 h, y_n + b_{21} h K_1)\\
# K_3 =& f(x_n + a_3 h, y_n + b_{31} h K_1 + b_{32} h K_2)
# \end{aligned}
# $$

# In[5]:


C1, C2, C3 = sp.symbols('C_1 C_2 C_3')
h, a2, a3 = sp.symbols('h a_2 a_3')
b21, b31, b32 = sp.symbols('b_{21} b_{31} b_{32}')


# In[6]:


K1 = f(x, y)
K2 = (f_taylor.subs([(dx, a2*h), (dy, b21*h*K1)])+sp.O(h**4)).expand().removeO()
K3 = (f_taylor.subs([(dx, a3*h), (dy, b31*h*K1 + b32*h*K2)])+sp.O(h**4)).expand().removeO()
ynp1 = (y_(x) + h * (C1*K1 + C2*K2 + C3*K3)).expand()
display(K1, K2, K3, ynp1)


# In[7]:


target_ynp1 = y_taylor.subs(dx, h).expand()
diff = ynp1 - target_ynp1
display(target_ynp1)
display(diff)


# In[8]:


params = [C1, C2, C3, h, a2, a3, b21, b31, b32]
conds = []
for p in range(1, 4):
    part_p = diff.coeff(h, p)
    for arg in part_p.args:
        tmp = 1
        for parts in arg.args:
            if not parts.is_number and not parts.has(*params):
                tmp *= parts
        cond = part_p.coeff(tmp, 1)
        if cond not in conds:
            conds.append(cond)
print(len(conds))
display(*conds)


# In[9]:


a = C2*(a2 - b21)**2/2 + C3*(a3 - b31 - b32)**2/2
b = conds[3] - conds[4] + conds[6]
display(a, b, a.expand()-b)


# In[10]:


conds_simp = [
    sp.Eq(0, a2 - b21),
    sp.Eq(0, a3 - b31 - b32),
    sp.Eq(0, conds[0]),
    sp.Eq(0, conds[1]),
    sp.Eq(0, conds[3]*2),
    sp.Eq(0, conds[7])
]
display(*conds_simp)


# ## Runge-Kutta-RK4

# In[11]:


import sympy as sp
def removeSub(expr):
    if isinstance(expr, sp.Subs):
        return expr.args[0].replace(expr.args[1][0], expr.args[2][0])
    for arg in expr.args:
        expr = expr.replace(arg, removeSub(arg))
    return expr


# In[12]:


x, dx, y, dy = sp.symbols('x \Delta{x} y \Delta{y}')
f, y_ = sp.symbols('f y', cls=sp.Function)


# In[13]:


dy_dx = f(x, y)
dy_dx2 = removeSub(f(x, y_(x)).diff(x).subs(y_(x).diff(x), dy_dx).subs(y_(x), y))
dy_dx3 = removeSub(f(x, y_(x)).diff(x, 2).subs(y_(x).diff(x, 2), dy_dx2).subs(y_(x).diff(x), dy_dx).subs(y_(x), y)).expand().simplify()
dy_dx4 = removeSub(f(x, y_(x)).diff(x, 3).subs(y_(x).diff(x, 3), dy_dx3).subs(y_(x).diff(x, 2), dy_dx2).subs(y_(x).diff(x), dy_dx).subs(y_(x), y)).expand().simplify()
y_taylor = y_(x) + dy_dx*dx + dy_dx2*dx**2/2 + dy_dx3*dx**3/6 + dy_dx4*dx**4/24
display(dy_dx, dy_dx2, dy_dx3, dy_dx4, y_taylor)


# In[14]:


pdf1 = f(x, y).diff(x)*dx + f(x, y).diff(y)*dy
pdf2 = (pdf1.diff(x)*dx + pdf1.diff(y)*dy).expand().simplify()
pdf3 = (pdf2.diff(x)*dx + pdf2.diff(y)*dy).expand().simplify()
f_taylor = f(x, y) + pdf1 + pdf2/2 + pdf3/6
display(pdf1, pdf2, pdf3, f_taylor)


# $$
# \begin{aligned}
# y_{n+1} =& y_n + h(C_1 K_1 + C_2 K_2 + C_3 K_3 + C_4 K_4)\\
# K_1 =& f(x_n, y_n)\\
# K_2 =& f(x_n + a_2 h, y_n + b_{21} h K_1)\\
# K_3 =& f(x_n + a_3 h, y_n + b_{31} h K_1 + b_{32} h K_2)\\
# K_4 =& f(x_n + a_4 h, y_n + b_{41} h K_1 + b_{42} h K_2 + b_{43} h k_2)
# \end{aligned}
# $$

# In[15]:


C1, C2, C3, C4 = sp.symbols('C_1 C_2 C_3 C_4')
h, a2, a3, a4 = sp.symbols('h a_2 a_3 a_4')
b21, b31, b32, b41, b42, b43 = sp.symbols('b_{21} b_{31} b_{32} b_{41} b_{42} b_{43}')


# In[16]:


K1 = f(x, y)
K2 = (f_taylor.subs([(dx, a2*h), (dy, b21*h*K1)])+sp.O(h**5)).expand().removeO()
K3 = (f_taylor.subs([(dx, a3*h), (dy, b31*h*K1 + b32*h*K2)])+sp.O(h**5)).expand().removeO()
K4 = (f_taylor.subs([(dx, a4*h), (dy, b41*h*K1 + b42*h*K2 + b43*h*K3)])+sp.O(h**5)).expand().removeO()
ynp1 = (y_(x) + h * (C1*K1 + C2*K2 + C3*K3 + C4*K4)).expand()


# In[8]:


target_ynp1 = y_taylor.subs(dx, h).expand()
diff = ynp1 - target_ynp1
display(target_ynp1)


# In[9]:


params = [C1, C2, C3, C4, h, a2, a3, a4, b21, b31, b32, b41, b42, b43]
conds = []
for p in range(1, 5):
    part_p = diff.coeff(h, p)
    for arg in part_p.args:
        tmp = 1
        for parts in arg.args:
            if not parts.is_number and not parts.has(*params):
                tmp *= parts
        cond = part_p.coeff(tmp, 1)
        if cond not in conds:
            conds.append(cond)
print(len(conds))
display(*conds)


# In[10]:


conds[3] - conds[4] + conds[6]


# In[11]:


a = C2*(a2 - b21)**2/2 + C3*(a3 - b31 - b32)**2/2 + C4*(a4 - b41 - b42 - b43)**2/2
b = conds[3] - conds[4] + conds[6]
display(a, b, a.expand() - b.expand())


# In[89]:


conds2 = [
    a2-b21,
    a3-b31-b32,
    a4-b41-b42-b43,
    conds[0],
    conds[1],
    conds[3]*2,
    conds[7],
    conds[8]*6,
    conds[10],
    conds[12],
    conds[15]*2,
]
print(len(conds2))
display(*conds2)


# In[87]:


conds[2].collect(C2).collect(C3).collect(C4).subs([(b21, a2), (b31+b32, a3), (b41+b42+b43, a4)])
conds[4].collect(C2*a2).collect(C3*a3).collect(C4*a4).subs([(b21, a2), (b31+b32, a3), (b41+b42+b43, a4)])
conds[6].collect(C3).collect(C4)
conds[5].subs(b21, a2).collect(C4*b43).subs(b31+b32, a3)
(conds[9]*2).subs(b21, a2).collect(C3*a3).collect(C4*a4)
(conds[11].collect(C3*a3**2/2).collect(C4*a4**2/2).subs([(b21, a2), (b31+b32, a3), (b41+b42+b43, a4)])*2)
conds[13].subs(b21, a2)
(conds[14]*6).subs(b21, a2).collect(C2).collect(C3).collect(C4)
conds[16].subs(b21, a2).collect(C4*a3*b43).collect(C4*a4*b43).subs(b31+b32, a3) - conds[15]*2 - conds[10]
conds[17].subs(b21, a2).collect(C3*a2*b32).collect(C4*b31*b43).collect(C4*b32*b43).collect(C4*a2*b42).subs([(b31+b32, a3), (b41+b42+b43, a4)]).expand().collect(C4*a4*b43).collect(C4*b43).subs([(b31+b32, a3), (((b31+b32)**2/2).expand(), a3**2/2)]).expand()-conds[10]-conds[15]
conds[18].collect(C3*a2*b32).collect(C4*a2*b42).collect(C4*a3*b43).subs([(b21, a2), (b31+b32, a3), (b41+b42+b43, a4)])


# $$
# \begin{aligned}
# y_{n+1} =& y_n + h(\frac{1}{6} K_1 + \frac{2}{6} K_2 + \frac{2}{6} K_3 + \frac{1}{6} K_4)\\
# K_1 =& f(x_n, y_n)\\
# K_2 =& f(x_n + \frac{h}{2}, y_n + \frac{h}{2} K_1)\\
# K_3 =& f(x_n + \frac{h}{2} h, y_n + \frac{h}{2} K_2)\\
# K_4 =& f(x_n + h, y_n + h k_2)
# \end{aligned}
# $$

# In[93]:


val_map = {
    C1:sp.Rational(1, 6), C2:sp.Rational(2, 6), C3:sp.Rational(2, 6), C4:sp.Rational(1, 6),
    a2:sp.Rational(1, 2), b21:sp.Rational(1, 2),
    a3:sp.Rational(1, 2), b31:sp.Rational(0, 1), b32:sp.Rational(1, 2),
    a4:sp.Rational(1, 1), b41:sp.Rational(0, 1), b42:sp.Rational(0, 1), b43:sp.Rational(1, 1)
}
for cond in conds2:
    display(cond.evalf(subs=val_map))

