import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


# user num = 50
labels = ['100', '120', '140', '160', '180', '200']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]



unl_vbu = [0.177, 0.199 , 0.240 , 0.287 , 0.310, 0.331]

unl_mbu = [0.208, 0.232, 0.273, 0.311, 0.349, 0.412]
unl_retrain = [1190, 1192, 1188, 1193, 1191, 1189]



for i in range(len(labels)):

    unl_retrain[i] = unl_retrain[i]
    unl_mbu[i] = unl_mbu[i]
    unl_vbu[i] = unl_vbu[i]


x = np.arange(len(labels))  # the label locations
width = 0.9 # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

plt.style.use('seaborn')
plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width /6 - width / 6 , unl_muv_MNIST, width=width/6, label='MUA-MD MNIST', color='#C6B3D3', edgecolor='black', hatch='/')

#plt.bar(x - width / 6 , unl_muv_CIFAR, width=width/6,  label='MUA-MD CIFAR10', color='#F7D58B', edgecolor='black' , hatch='x')
#E58579, 80BA8A


# F7D58B, 9CD1C8, C6B3D3, E58579
plt.bar(x - width / 4  , unl_vbu,   width=width/4, label='VBU', color='#61DE45', edgecolor='black',  hatch='x')

# F7D58B , 6BB7CA
plt.bar(x , unl_mbu, width=width/4, label='TCU', color='#F7D58B', edgecolor='black', hatch='*')

plt.bar(x + width / 4 , unl_retrain, width=width/4, label='Retrain', color='#C6B3D3', edgecolor='black', hatch='o')


#plt.bar(x + width / 6 + width / 6 + width/6  , unl_mib_CelebA,   width=width/6, label='MIB CelebA', color='#E58579', edgecolor='black', hatch='\\')
# plt.bar(x - width / 8 - width / 16, unl_vib, width=0.168, label='PriMU$_{w}$', color='cornflowerblue', hatch='*')
# plt.bar(x + width / 8, unl_self_r, width=0.168, label='PriMU$_{w/o}$', color='g', hatch='x')
# plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBFU', color='orange', hatch='\\')


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 2000, 400)
# Make y-axis a log scale:
plt.yscale('log')
# plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.gca().yaxis.get_major_formatter().set_scientific(True)
# plt.gca().yaxis.get_major_formatter().set_useOffset(False)

# plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)
# plt.grid(axis='y')
# plt.legend(loc='upper left', fontsize=20)

plt.legend( frameon=True, facecolor='#EAEAF2', loc='center', bbox_to_anchor=(0.52001, -0.21), ncol=3, fontsize=14.6,)

# mode="expand",  columnspacing=1.0,  borderaxespad=0., framealpha=0.5,handletextpad=0.5
#title = 'Methods and Datasets',

plt.xlabel('$\it{USS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar10_running_time_cs_as_bar.pdf', format='pdf', dpi=200)
plt.show()
