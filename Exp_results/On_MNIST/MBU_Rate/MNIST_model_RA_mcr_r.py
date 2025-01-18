

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.001', '0.002', '0.004', '0.006', '0.008', '0.01']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.9955, 0.9954, 0.9952, 0.9950, 0.9948, 0.9946]

org_acc = [0.9960, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960]

vbu_acc = [0.9442, 0.9445,  0.9401, 0.9411, 0.9472, 0.9467]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
VBU = [0.9937, 0.9937, 0.9937, 0.9937, 0.9937, 0.9937]

for i in range(len(OUL)):
    OUL[i] = OUL[i]*100
    org_acc[i] = org_acc[i]*100
    vbu_acc[i] = vbu_acc[i]*100
    VBU[i] = VBU[i] * 100

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='TCU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




# plt.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='GA',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, VBU, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
         label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Remaining Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(99.2, 99.8, 0.1)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\alpha$' ,fontsize=20)

plt.xticks(x, labels, fontsize=18)
# plt.title('CIFAR10 IID')

# plt.annotate(r"1e0", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=(0.53, 0.01),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('MNIST_model_RA__mbu_r.pdf', format='pdf', dpi=200)
plt.show()