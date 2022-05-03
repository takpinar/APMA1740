import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

x = np.linspace(-10, 13, 1200)

####################################################
fig1, ax1 = plt.subplots()
#Plot f_1(x)
ax1.plot(x, norm(loc=1, scale=1).pdf(x), label='F_1(x)')

#plot f_2(x)
ax1.plot(x, norm(loc=2, scale=4).pdf(x), label='F_2(x)')

#plot f(x)
ax1.plot(x, norm(loc=(.45+2*.55), scale=(.45+4*.55)).pdf(x), lw=3, label='F(x)')

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
r2 = 0.55*norm(loc=2, scale=4).pdf(x)
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







