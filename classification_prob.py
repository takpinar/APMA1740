import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import random

x = np.linspace(-4, 8, 1200)

####################################################
#1)
####################################################
fig1, ax1 = plt.subplots()
#Plot f_1(x)
ax1.plot(x, norm(loc=1, scale=1).pdf(x), label='F_1(x)')

#plot f_2(x)
ax1.plot(x, norm(loc=2, scale=2).pdf(x), label='F_2(x)')

#plot f(x)
ax1.plot(x, norm(loc=(.45+2*.55), scale=(.45+2*.55)).pdf(x), lw=3, label='F(x)')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax1.legend(fontsize=23)
fig1.set_size_inches(20, 10.5)
fig1.savefig('3gaussians.png')
plt.show()

####################################################

fig2, ax2 = plt.subplots()

#plot r_1(x)
r1 = 0.45*norm(loc=1, scale=1).pdf(x)
ax2.plot(x, r1, label='r_1(x)')

#plot r_1(x)
r2 = 0.55*norm(loc=2, scale=2).pdf(x)
ax2.plot(x, r2, label='r_2(x)')
ax2.legend(fontsize=23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig2.set_size_inches(20, 10.5)
fig2.savefig('r_graphs.png')
plt.show()

####################################################
fig3, ax3 = plt.subplots()
ax3.plot(x, r1, label='r_1(x)')
ax3.plot(x, r2, label='r_2(x)')
ax3.fill_between(x,0,.2, where=r2>r1, color='green', alpha=0.2, label='Set B')
ax3.legend(fontsize=23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig3.set_size_inches(20, 10.5)
fig3.savefig('SetB.png')
plt.show()

####################################################
#2)
####################################################

f,a = plt.subplots()
plt.plot(x, norm(loc=1, scale=1).pdf(x), label='N(1,1)')

plt.plot(x, norm(loc=2, scale=2).pdf(x), label='N(2,4)')

plt.fill_between(x, 0, norm(loc=1, scale=1).pdf(x), where=x>1.5, color='blue', 
    alpha=0.1, label='A = {}'.format(1-norm(loc=1, scale=1).cdf(1.5)))

plt.fill_between(x, 0, norm(loc=2, scale=2).pdf(x), where=x<=1.5, color='orange', 
    alpha=0.19, label='A = {}'.format(norm(loc=2, scale=2).cdf(1.5)))

plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
f.set_size_inches(20, 10.5)
f.savefig('cdfs1p5.png')
plt.show()

####################################################
def h_1p5(x):
    if x>1.5:
        return 2
    else:
        return 1

data=[]
for _ in range(10**6):

    rand = random.random()
    if rand<0.45:
        x = np.random.normal(1,1)
        y=1
    else:
        x = np.random.normal(2,4)
        y=2
    data.append([x,y])

correct = 0
for d in data:
    if h_1p5(d[0])==d[1]:
        correct+=1

errRate = 1 - (correct/10**6)
print('t=1.5 Error Rate = {}'.format(errRate))

#################################
x = np.linspace(-4, 8, 1200)
f1,a1 = plt.subplots()
plt.plot(x, norm(loc=1, scale=1).pdf(x), label='N(1,1)')

plt.plot(x, norm(loc=2, scale=2).pdf(x), label='N(2,4)')

plt.fill_between(x, 0, norm(loc=1, scale=1).pdf(x), where=x>2.5, color='blue', 
    alpha=0.1, label='A = {}'.format(1-norm(loc=1, scale=1).cdf(2.5)))

plt.fill_between(x, 0, norm(loc=2, scale=2).pdf(x), where=x<=2.5, color='orange', 
    alpha=0.19, label='A = {}'.format(norm(loc=2, scale=2).cdf(2.5)))

plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
f1.set_size_inches(20, 10.5)
f1.savefig('cdfs2p5.png')
plt.show()


##################################################################

f2, a2 = plt.subplots()
#Plot f_1(x)
a2.plot(x, norm(loc=1, scale=1).pdf(x), label='F_1(x)')

a2.plot(x, norm(loc=2, scale=2).pdf(x), label='F_2(x)')
A = norm(2,2).cdf(1.99)-norm(2,2).cdf(-0.659)
plt.fill_between(x, 0, norm(loc=2, scale=2).pdf(x), where=(x>-0.659)&(x<1.99), color='orange', alpha=0.19,label='A = {}'.format(A))
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
f2.set_size_inches(20, 10.5)
f2.savefig('probNot2.png')
plt.show()

###################
f3, a3 = plt.subplots()

a3.plot(x, norm(loc=1, scale=1).pdf(x), label='F_1(x)')

a3.plot(x, norm(loc=2, scale=2).pdf(x), label='F_2(x)')
A = norm(1,1).cdf(-0.659) + (1-norm(1,1).cdf(1.99))
plt.fill_between(x, 0, norm(loc=1, scale=1).pdf(x), where=(x<-0.659), alpha=0.1, color='blue', label='A = {}'.format(A))
plt.fill_between(x, 0, norm(loc=1, scale=1).pdf(x), where=(x>1.99), alpha=0.1, color='blue')

plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
f3.set_size_inches(20, 10.5)
f3.savefig('probNot1.png')
plt.show()



