

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['200', '400', '600', '800', '1000', '1200']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]


OUL_finetuned = [0.9961, 0.9945, 0.9951, 0.9951, 0.9955, 0.9950]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
OUL = [0.9958, 0.9948, 0.9947, 0.9943, 0.9941, 0.9940]

org_acc = [0.9965, 0.9965, 0.9965, 0.9965, 0.9965, 0.9965]

salun_acc = [0.9937, 0.9938,  0.9941, 0.9934, 0.9938, 0.9935]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]
vbu_ldp_acc = [0.9937, 0.9933, 0.9930, 0.9929, 0.9924, 0.9922]

for i in range(len(OUL)):
    OUL[i] = OUL[i]*100
    org_acc[i] = org_acc[i]*100
    salun_acc[i] = salun_acc[i]*100
    vbu_ldp_acc[i] = vbu_ldp_acc[i] * 100
    OUL_finetuned[i] = OUL_finetuned[i] * 100

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Origin',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='TCU-S', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



plt.plot(x, salun_acc, linestyle='-.', color='#B595BF',  marker='d', fillstyle='full', markevery=markevery, label='SalUn',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, vbu_ldp_acc, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
         label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, OUL_finetuned, linestyle='-.', color='#E07E35',  marker='*', fillstyle='full', markevery=markevery, label='TCU-S(FineTuned)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Remaining Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(98.8, 99.9, 0.2)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\it USS$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

# plt.annotate(r"1e0", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=(0.03, 0.01),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('MNIST_model_RA_uss_finetuned.pdf', format='pdf', dpi=200)
plt.show()