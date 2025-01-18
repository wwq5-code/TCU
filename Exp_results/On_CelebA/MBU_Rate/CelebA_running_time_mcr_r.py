

import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
# data construction time
OUL = [0.657479+1.60, 0.640489+1.65, 0.697980165+1.68, 0.72441936+1.60, 0.7297673810+1.91, 0.96332+2.08]

org_acc = [654.3272, 654.3272, 648.3272, 648.3272, 665.3272, 663.3272]

vbu_acc = [0.672800, 0.72140, 0.92958, 0.957286, 1.1322344, 1.1822344]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]


for i in range(len(OUL)):
    OUL[i]=OUL[i]*1
    org_acc[i]=org_acc[i]
    vbu_acc[i]=vbu_acc[i]*1

plt.style.use('seaborn')
#plt.figure(figsize=(5.5, 5.3))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 5.3), gridspec_kw={'height_ratios': [1, 3]})

# Create a broken axis instance
#bax = brokenaxes(ylims=((0, 2), (650, 670)), hspace=.15)


l_w=5
m_s=15
marker_s=3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)


# Upper plot (y-axis range: 200-300)
ax1.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Retrain',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax1.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery, label='TCU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax1.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='GA',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax1.set_ylim(600, 700)
ax1.tick_params(axis='y', labelsize=16)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)




# Lower plot (y-axis range: 0-50)
ax2.plot(x, org_acc, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery, label='Retrain',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax2.plot(x, OUL, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery, label='MBU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax2.plot(x, vbu_acc, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='GA',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

ax2.set_ylim(0, 5)
ax2.tick_params(axis='y', labelsize=16)
ax2.spines['top'].set_visible(False)

# Add diagonal lines to indicate a break in the y-axis
d = .015  # size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# Labels
ax2.set_xlabel('$\\beta$' ,fontsize=20)
ax2.set_ylabel('Running Time (s)', fontsize=24)
# ax2.yaxis.set_label_position('right')
ax2.yaxis.set_label_coords(-0.1, 0.7)
ax2.set_xticks(x, labels, fontsize=20)
ax2.legend(loc='best',fontsize=20)
# plt.grid()
#leg = bax.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
#bax.set_ylabel('Running Time (s)', fontsize=24)


#my_y_ticks = np.arange(0., 300.94, 50.)
#plt.yticks(my_y_ticks,fontsize=20)
#bax.set_xlabel('$\\it USR$ (%)' ,fontsize=20)


# plt.title('CIFAR10 IID')

#plt.annotate(r"1e-1", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10), textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
#bax.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('CelebA_running_time_mcr_r.pdf', format='pdf', dpi=200)
plt.show()