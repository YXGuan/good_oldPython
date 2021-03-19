#!/usr/bin/env python
# coding: utf-8

# In[1]:


raise(NameError)
    print('done')
func(10)

#Q3:
x=6
y=0
x>=2 and y!=0 and (x/y)>2
#x>=2 and (x/y)>2 and y!=0 

#6
def sum(a,b):
    s = []
    for i in range( len(a)):
s.append(a[i]+ b[i])
    return s

def add(A,B):
    c = []
    for i in range(len(A)):
c.append( sum(A[i],B[i]))
    return c

A = [[2,1,3],[-1,2,0]]
B = [[1,1,-1],[2,0,6]]

print(add(A,B))

def a(n1,n2):
    g = 1
    k = 2
    while k<=n1 and k<= n2:
if n1%k == 0 and n2%k == 0:
    g = k
k+= 1
    return g
print(a(4,12))


print(38%6)

product = 1
#for d in range(2,7,2):
#product =  1

def f1 (x,y):
    return x+y
def f2 (x,y,f):
    return f1(x,y)
def f3(x,y,f):
    return f(x,y,f)
print(f3(1,1,f2))

print("-----------------------------------")
x = 4 
for a in range(1, x+1):
    for b in range (1, a+1):
print(a*b, end  = ' ' )
    print()
print("-----------------------------------")
sum = 0
i = 1
#while i <10:
#    sum = sum+1
#i = i +1
#print(i)

def b(age):
    phrase = ' '
    if(age<16):
print('0')
    if(age>=16):
print('1')
    if(age>=18):
print('2')
    if(age>=19):
print('3')

b(19)

#26
value = 12345678.926
#print("the value is : {0:9.2 f}".format(value))

#59
def q(x):
    if x**2 <= 100:
if x% 4 == 2:
    return "six"
elif not x+ 5 >2:
    return x - 50
    else:
print("asdf")
print(q(9.0))





#s= input("enter a string")

#print(s)


#55
import math
radius= -20

if radius >= 0:
    area = radius* radius *math.pi
#print("ther area is :", area)

#47
print("____________________________")
a = []
for  i in range(5):
    b = []
    for j in range (i+1):
b.append(j)
    a.append(b)
print(a)


a = 1
def b(c):
    d = a + c
    return d

b(2)



#45 
import random
print(random.randrange(0,1))

#44
3/4.0*math.pi*10**-1*math.sqrt(2)
3*math.pi/4*10**-1*math.sqrt(2)


#42
print(range(0,11))


#41 s = 'Eng1D04'
print(s.lower())
#s[3] = '2'
print(s[0])
print(s.strip())

ini="ENG 1D04"
c = ini.index("4")
print(ini[c-2:])
c = ini.index("D")
print(ini[c:])

def a(x):
    try:
a = x
b = x[:]
return a/b
    except ZeroDivisionError:
return 1
    except TypeError:
return 3
    
L = []
print(a(L))


#x, y = eval(input("Enter two numbers"))


def a (s,b):
    n = len(s)
    d = 1
    f = 0
    for i in range (n):
f = f + int(s[n-i-1])*d
d = d*b
    return f
print(a('1111',2))


print("12345678.92"[9.2])




"asdfjh,asdfk".split(,)


# In[ ]:




