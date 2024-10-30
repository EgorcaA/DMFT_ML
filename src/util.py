
import numpy as np
from triqs_maxent import *

from h5 import *
import numpy as np
from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt
plt.rcParams['text.usetex'] = True


def findExtrema(inputArray):
   arrayLength = len(inputArray)
   outputCount = 0
   for k in range(1, arrayLength - 1):
      outputCount += (inputArray[k] > inputArray[k - 1] and inputArray[k] > inputArray[k + 1])
      outputCount += (inputArray[k] < inputArray[k - 1] and inputArray[k] < inputArray[k + 1])
   return outputCount

def gaussian(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def get_Gw(G_tau):
   tm = TauMaxEnt(cost_function='bryan', probability='normal')
   tm.omega = HyperbolicOmegaMesh(omega_min=-1.5, omega_max=1.5, n_points=101)
   tm.set_G_tau(G_tau)
   err = 5.e-4
   tm.set_error(err)
   result = tm.run()
   A0 = result.analyzer_results['Chi2CurvatureAnalyzer']['A_out']
   A0ens = result.omega
   return A0ens, A0 


def predict_G(sample):
    gl = np.real(sample.G0_l.data).flatten()
    beta = sample.beta
    U = sample.U
    features = np.hstack([U, beta, gl])
    leg = MLmodel(torch.Tensor(features)).cpu().detach().numpy()

    meshQ = MeshLegendre(beta, 'Fermion', 30)
    Gpredicted = Gf(mesh=meshQ, target_shape=[1,1])
    Gpredicted.data[:] = leg[:, np.newaxis, np.newaxis]

    tau_mesh = MeshImTime(beta=beta, statistic='Fermion', n_tau=sample.n_tau)
    GtauML = Gf(mesh=tau_mesh, target_shape=[1,1])
    GtauML.set_from_legendre(Gpredicted)

    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=sample.n_iw)
    G_iw_ML = Gf(mesh=iw_mesh, target_shape=[1,1])
    G_iw_ML.set_from_legendre(Gpredicted)

    return G_iw_ML, GtauML, Gpredicted #leg

# G_iw_ML, GtauML, Gpredicted = predict_G(s)

def plot_all(sample,  A0ens, A0, Aens, A, Aens_ML, A_ML, sigma=0.2):

    G_iw_ML, GtauML, Gl_ML = predict_G(sample)


    fig = plt.figure(constrained_layout=True, figsize=( 4, 4))
    # axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
    #                         gridspec_kw={'width_ratios':[2, 1]})
    
    axs = fig.subplot_mosaic([['Top', 'Top'],['BottomLeft', 'BottomRight']],
                            gridspec_kw={'width_ratios':[2, 1]})
    # axs['Top'].set_title('Spectral function')


    energy_min = -3  # Minimum energy value
    energy_max = 3  # Maximum energy value
    n_points = 100   # Number of points in the energy grid
    energy_grid = np.linspace(energy_min, energy_max, n_points)

    dos = np.zeros_like(energy_grid)

    # for energy in sample.BS.flatten():
    #     dos += gaussian(energy_grid, energy, sigma)
    # dos_integral = np.trapz(dos, energy_grid)  # Compute the integral of the DOS
    # dos /= dos_integral  # Normalize the DOS
    # axs['Top'].plot(energy_grid, dos, label=f'Gaussian Broadened DOS (sigma={sigma})')

    # A0ens, A0 =  get_Gw(sample.G0_tau)
    axs['Top'].plot(A0ens, A0, 'b', label=r'-$\frac{1}{\pi}$ Im $G_0$')
    
    # Aens, A =  get_Gw(sample.G_tau)
    axs['Top'].plot(Aens, A, 'orange', label=r'-$\frac{1}{\pi}$ Im $G$')

    # Aens_ML, A_ML =  get_Gw(GtauML)
    axs['Top'].plot(Aens_ML, A_ML, 'r', label=r'-$\frac{1}{\pi}$ Im $G_{ML}$')

    axs['Top'].set_xlim((-1, 1))
    axs['Top'].set_ylim((0, max(np.max(A0)*1.5,  np.max(A0)*1.5) ))
    axs['Top'].set_xlabel(r'$E - E_f$(eV)')
    axs['Top'].set_ylabel('Density of States (DOS)')
    axs['Top'].text(-0.13, 1.0, 'a)', transform=axs['Top'].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    axs['Top'].legend(prop={'size': 10}, frameon=False)  

    # plt.legend()

    #################################################################################################    
    # axs['BottomLeft'].set_title('Plot Top Right')

    ntau = 200
    taumesh = np.linspace(0, sample.beta, ntau)
    G0_rebinned = sample.G0_tau.rebinning_tau(new_n_tau=ntau)
    G_rebinned = sample.G_tau.rebinning_tau(new_n_tau=ntau)
    GML_rebinned = GtauML.rebinning_tau(new_n_tau=ntau)


    axs['BottomLeft'].plot(taumesh, np.real(G_rebinned.data.flatten()),  'orange', label='G')
    axs['BottomLeft'].plot(taumesh, np.real(G0_rebinned.data.flatten()), 'b', label='G0')
    axs['BottomLeft'].plot(taumesh, np.real(GML_rebinned.data.flatten()), 'r', label='G ML')

    axs['BottomLeft'].set_xlabel(r'$\tau$(eV)')
    axs['BottomLeft'].set_ylabel(r'$G(\tau)$ (eV$^{-1}$)')
    axs['BottomLeft'].set_xlim((0, sample.beta))
    
    axs['BottomLeft'].text(-0.2, 1.0, 'b)', transform=axs['BottomLeft'].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    #################################################################################################    
    # axs['BottomRight'].set_title('Legendre')
    xs = np.arange(0, 30, 1)
    axs['BottomRight'].scatter(xs, np.real(sample.G0_l.data.flatten())[:30], c='b', marker='D', label='G0')
    axs['BottomRight'].scatter(xs, np.real(sample.G_l.data.flatten())[:30], c='orange',  marker='s', label='G')
    axs['BottomRight'].scatter(xs, np.real(Gl_ML.data.flatten())[:30], c='r',  marker='x', label='G ML')

    axs['BottomRight'].set_xlabel('n')
    axs['BottomRight'].set_ylabel(r'$G_l$')
    axs['BottomRight'].set_xlim((0, 10))
    
    
    axs['BottomRight'].text(-0.2, 1.0, 'c)', transform=axs['BottomRight'].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    
    plt.savefig('./sample.pdf', dpi=200, bbox_inches='tight')

    plt.show()



def plot_DMFT_tau(filename='data_sc.h5'):
    with HDFArchive('./data/' + filename, 'r') as a: ps = a['ps']
    p = ps.objects[-1]
        
    plt.figure(figsize=(5, 5))
    
    for p in ps.objects:
        oplotr(p.G_tau)
    
        # plt.plot(np.arange(1, len(G0_l), 1), G0_l, 's-')
    
    plt.show()


# def plot_DMFT_compare():
#     with HDFArchive('./data/data_sc_true.h5', 'r') as a: ps = a['ps']
#     p = ps.objects[-1]

#     with HDFArchive('./data/data_sc_ML.h5', 'r') as a: ds = a['ps']
#     d = ds.objects[-1]
        
#     fig, axes = plt.subplots(2, 2, figsize=(5, 5))

#     # plt.figure(figsize=(5, 5))
#     # subp = [2, 2, 1]

#     axes[0,0].plot(ps.iter-1, ps.dG_l, 's-')
#     axes[0,0].plot(ds.iter-1, ds.dG_l, 's-')
#     axes[0,0].set_ylabel('$\max | \Delta G_l |$')
#     axes[0,0].set_xlabel('Iteration')
#     axes[0,0].semilogy([], [])
#     axes[0, 0].set_xlim((1, 10))
#     axes[0,0].text(-0.3, 1.0, 'a)', transform=axes[0,0].transAxes,
#         fontsize=10, fontweight='normal', va='top', ha='right')
    
#     # for b, g in p.G_l: p.G_l[b].data[:] = np.abs(g.data)
#     # print(d.G_l.data)
#     axes[0, 1].plot(np.arange(0, 30, 1), np.abs(p.G_l['up'].data.flatten())[:30], 'o-', markersize=4, linewidth=0.8)
#     axes[0, 1].plot(np.arange(0, 30, 1), np.abs(d.G_l.data.flatten()),            'o-', markersize=4, linewidth=0.8)
    
#     axes[0, 1].set_ylabel('$| G_l |$')
#     axes[0, 1].semilogy([], [])
#     axes[0,1].text(-0.2, 1.0, 'b)', transform=axes[0,1].transAxes,
#         fontsize=10, fontweight='normal', va='top', ha='right')
    
#     axes[1, 0].plot(np.linspace(0, 1, len(p.g_tau.data.flatten())), p.g_tau.data.flatten(), markersize=8, linewidth=2)
#     axes[1, 0].plot(np.linspace(0, 1, len(d.g_tau.data.flatten())), d.g_tau.data.flatten(), markersize=8, linewidth=2)
#     # axes[0, 1].set_legend(loc='best', fontsize=8)
#     axes[1, 0].set_ylabel(r'$G(\tau)$')
#     axes[1, 0].set_xlabel(r'$\tau/\beta$')
#     axes[1, 0].set_xlim((0, 1))
#     axes[1,0].text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
#         fontsize=10, fontweight='normal', va='top', ha='right')
    
#     # oploti(p.sigma_w[0,0], '.-' )
#     # oploti(p.sigma_w, '.-')
#     ll = len(np.imag(d.sigma_w.data.flatten()))//2
#     axes[1, 1].plot(np.arange(-ll, ll, 1), np.imag(p.sigma_w.data.flatten()), '.-' )
#     axes[1, 1].plot(np.arange(-ll, ll, 1), np.imag(d.sigma_w.data.flatten()), '.-' )

#     axes[1, 1].set_ylabel(r'$\Sigma(i\omega_n)$')
#     axes[1, 1].set_xlim([0, 50])
#     axes[1, 1].set_ylim([-1, 0])
#     axes[1, 1].set_xlabel(r'$i\omega_n$')
#     axes[1, 1].text(-0.2, 1.0, 'd)', transform=axes[1, 1].transAxes,
#         fontsize=10, fontweight='normal', va='top', ha='right')
    
#     plt.tight_layout()
#     # plt.savefig('./pics/DMFT_sample.pdf' , dpi=200, bbox_inches='tight')
#     plt.show()


def plot_DMFT_compare(ps, ds):
    # with HDFArchive('./data/data_sc_true.h5', 'r') as a: ps = a['ps']
    p = ps[-1] # exact

    # with HDFArchive('./data/data_sc_ML.h5', 'r') as a: ds = a['ps']
    d = ds[-1]# ML
        
    beta = ps[-1].init.beta
    fig, axes = plt.subplots(2, 2, figsize=(5, 4))

    # plt.figure(figsize=(5, 5))
    # subp = [2, 2, 1]
    iterr = [x.iter -1 for x in ps]
    dG_l = [x.dG_l for x in ps]
    axes[0,0].plot(iterr, dG_l, 's-', c='b')
    iterr = [x.iter-1 for x in ds]
    dG_l = [x.dG_l for x in ds]
    axes[0,0].plot(iterr, dG_l, 's-', c='r')
    
    axes[0,0].set_ylabel('$\max | \Delta G_l |$')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].semilogy([], [])
    # axes[0, 0].set_xlim((1, 10))
    axes[0,0].text(-0.3, 1.0, 'a)', transform=axes[0,0].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    # for b, g in p.G_l: p.G_l[b].data[:] = np.abs(g.data)
    # print(d.G_l.data)
    axes[0, 1].plot(np.arange(0, 30, 1), np.abs(p.g_l.data.flatten())[:30], 'o-', color='b', markersize=4, linewidth=0.8)
    axes[0, 1].plot(np.arange(0, 30, 1), np.abs(d.g_l.data.flatten()),      'o-', color='r', markersize=4, linewidth=0.8)
    
    axes[0, 1].set_ylabel('$| G_l |$')
    axes[0, 1].semilogy([], [])
    axes[0,1].text(-0.2, 1.0, 'b)', transform=axes[0,1].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    axes[1, 0].plot(np.linspace(0, 1, len(p.g_tau.data.flatten())), p.g_tau.data.flatten(),c='b', markersize=8, linewidth=2)
    axes[1, 0].plot(np.linspace(0, 1, len(d.g_tau.data.flatten())), d.g_tau.data.flatten(),c='r', markersize=8, linewidth=2)
    # axes[0, 1].set_legend(loc='best', fontsize=8)
    axes[1, 0].set_ylabel(r'$G(\tau)$')
    axes[1, 0].set_xlabel(r'$\tau/\beta$')
    axes[1, 0].set_xlim((0, 1))
    axes[1,0].text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    # oploti(p.sigma_w[0,0], '.-' )
    # oploti(p.sigma_w, '.-')
    ss = np.imag(d.sigma_w.data.flatten())
    ll = len(np.imag(ss))//2

    axes[1, 1].plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(p.sigma_w.data.flatten()), '.-', color='b' )
    axes[1, 1].plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(d.sigma_w.data.flatten()), '.-', color='r' )

    axes[1, 1].set_ylabel(r'$\Sigma(i\omega_n)$ [eV]')
    axes[1, 1].set_xlim([0, 10])
    # print(-1.5*np.min(ss))
    axes[1, 1].set_ylim([1.1*np.min(ss[ll:]), 0])
    axes[1, 1].set_xlabel(r'$i\omega_n$ [eV]')
    axes[1, 1].text(-0.2, 1.0, 'd)', transform=axes[1, 1].transAxes,
        fontsize=10, fontweight='normal', va='top', ha='right')
    
    plt.tight_layout()
    plt.savefig('./pics/DMFT_compare.pdf' , dpi=200, bbox_inches='tight')
    plt.show()


def plot_Sigma_Ex_ML_Both(ps_Exact, ps_ML, extra=None):
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0,1,200))


    fig, ax = plt.subplots( figsize=(5, 3))

    # plt.figure(figsize=(5, 5))
    # subp = [2, 2, 1]
    # ll = len(ps_Exact[-1].sigma_w.data.flatten())
    ss = np.imag(ps_ML[-1].sigma_w.data.flatten())
    ll = len(ss)//2
    beta = ps_Exact[-1].init.beta
    for iter, p in enumerate(ps_Exact):
        ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(p.sigma_w.data.flatten()), 
                markersize=8, linewidth=2, label=f'Iter {(iter+1):.0f}')



    ax.set_prop_cycle(color=cmap(np.linspace(0,1,20)))
    # plt.colorbar.ColorbarBase(ax, cmap = colors, orientation='vertical')

    ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(ps_ML[-1].sigma_w.data.flatten()), 'o-',
            markersize=8, linewidth=2, c='blue', label='ML')

    # ax.plot(np.arange(-ll, ll, 1), np.imag(Sigma5), markersize=8, linewidth=2, label='1 iter')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaML), markersize=8, linewidth=2, label='ML')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaBoth), markersize=8, linewidth=2, label='ML + 1 iter')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.set_ylabel(r'$\Sigma(i\omega_n)$')
    ax.set_xlim([0, 10])
    ax.set_ylim([np.min(ss[ll:])*1.1, 0])
    ax.set_xlabel(r'$i\omega_n$ [eV]')
    # ax.text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
    #     fontsize=10, fontweight='normal', va='top', ha='right')
    
    # oploti(p.sigma_w[0,0], '.-' )
    # oploti(p.sigma_w, '.-')
    # plt.savefig('figure_sc.svg')
    plt.savefig('./pics/DMFT_sigma.pdf' , dpi=200, bbox_inches='tight')


    plt.show()


def plot_Sigma_Ex_ML_v3v4(ps_Exact, ps_ML_v3, ps_ML_v4, extra=None):
    cmap = plt.get_cmap('hsv')
    # colors = cmap(np.linspace(0,1,200))

    cmap = plt.cm.Blues
    num_colors = 3


    fig, ax = plt.subplots( figsize=(5, 3))

    # plt.figure(figsize=(5, 5))
    # subp = [2, 2, 1]
    # ll = len(ps_Exact[-1].sigma_w.data.flatten())
    ss = np.imag(ps_ML_v4[-1].sigma_w.data.flatten())
    ll = len(ss)//2
    beta = ps_Exact[-1].init.beta
    for iter, p in enumerate(ps_Exact):
        color = cmap((iter + 3 )/ (num_colors+ 3 ))
        ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(p.sigma_w.data.flatten()), 
                color=color, markersize=8, linewidth=2, label=f'Iter {(iter+1):.0f}')



    ax.set_prop_cycle(color=cmap(np.linspace(0,1,20)))
    # plt.colorbar.ColorbarBase(ax, cmap = colors, orientation='vertical')

    ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(ps_ML_v4[-1].sigma_w.data.flatten()), 'o-',
            markersize=4, linewidth=2, c='red', label='ML int. steps')

    ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.imag(ps_ML_v3[-1].sigma_w.data.flatten()), 'o-',
            markersize=4, linewidth=2, c='brown', label='ML one-shot')


    # ax.plot(np.arange(-ll, ll, 1), np.imag(Sigma5), markersize=8, linewidth=2, label='1 iter')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaML), markersize=8, linewidth=2, label='ML')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaBoth), markersize=8, linewidth=2, label='ML + 1 iter')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.set_ylabel(r'$\Sigma(i\omega_n)$')
    ax.set_xlim([0, 10])
    ax.set_ylim([np.min(ss[ll:])*1.3, 0])
    ax.set_xlabel(r'$i\omega_n$ [eV]')
    # ax.text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
    #     fontsize=10, fontweight='normal', va='top', ha='right')
    
    # oploti(p.sigma_w[0,0], '.-' )
    # oploti(p.sigma_w, '.-')
    # plt.savefig('figure_sc.svg')
    plt.savefig('./pics/DMFT_sigma_v2.pdf' , dpi=200, bbox_inches='tight')


    plt.show()



def plot_Sigma_Re(ps_Exact, ps_ML, extra=None):
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0,1,200))


    fig, ax = plt.subplots( figsize=(5, 3))

    # plt.figure(figsize=(5, 5))
    # subp = [2, 2, 1]
    # ll = len(ps_Exact[-1].sigma_w.data.flatten())
    ss = np.imag(ps_ML[-1].sigma_w.data.flatten())
    ll = len(ss)//2
    beta = ps_Exact[-1].init.beta
    for iter, p in enumerate(ps_Exact):
        ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.real(p.sigma_w.data.flatten()), 
                markersize=8, linewidth=2, label=f'Iter {(iter+1):.0f}')



    ax.set_prop_cycle(color=cmap(np.linspace(0,1,20)))
    # plt.colorbar.ColorbarBase(ax, cmap = colors, orientation='vertical')

    ax.plot(np.arange(-ll, ll, 1)*6.28/beta, np.real(ps_ML[-1].sigma_w.data.flatten()), 'o-',
            markersize=8, linewidth=2, c='blue', label='ML')

    # ax.plot(np.arange(-ll, ll, 1), np.imag(Sigma5), markersize=8, linewidth=2, label='1 iter')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaML), markersize=8, linewidth=2, label='ML')
    # ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaBoth), markersize=8, linewidth=2, label='ML + 1 iter')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.set_ylabel(r'$\Sigma(i\omega_n)$')
    ax.set_xlim([0, 10])
    # ax.set_ylim([np.min(ss[ll:])*1.1, 0])
    ax.set_xlabel(r'$i\omega_n$ [eV]')
    # ax.text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
    #     fontsize=10, fontweight='normal', va='top', ha='right')
    
    # oploti(p.sigma_w[0,0], '.-' )
    # oploti(p.sigma_w, '.-')
    # plt.savefig('figure_sc.svg')
    # plt.savefig('./pics/DMFT_sigma.pdf' , dpi=200, bbox_inches='tight')


    plt.show()




# def plot_Sigma_Ex_ML_Both(SigmaEx,Sigma5, SigmaML, SigmaBoth):
        
#     fig, ax = plt.subplots( figsize=(5, 5))

#     # plt.figure(figsize=(5, 5))
#     # subp = [2, 2, 1]
#     ll = len(SigmaEx)//2
#     ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaEx), markersize=8, linewidth=2, label='Exact')
#     ax.plot(np.arange(-ll, ll, 1), np.imag(Sigma5), markersize=8, linewidth=2, label='1 iter')
#     ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaML), markersize=8, linewidth=2, label='ML')
#     ax.plot(np.arange(-ll, ll, 1), np.imag(SigmaBoth), markersize=8, linewidth=2, label='ML + 1 iter')
#     ax.legend(loc='best', fontsize=8)
#     ax.set_ylabel(r'$\Sigma(i\omega_n)$')
#     ax.set_xlim([0, 50])
#     ax.set_ylim([-1, 0])
#     ax.set_xlabel(r'$i\omega_n$')
#     # ax.text(-0.3, 1.0, 'c)', transform=axes[1,0].transAxes,
#     #     fontsize=10, fontweight='normal', va='top', ha='right')
    
#     # oploti(p.sigma_w[0,0], '.-' )
#     # oploti(p.sigma_w, '.-')
#     plt.show()


# def plot_DMFT_test(ps):
#     p = ps[-1]
        
#     plt.figure(figsize=(3.25*2, 5))
#     subp = [2, 2, 1]

#     plt.subplot(*subp); subp[-1] += 1

#     iterr = [x.iter for x in ps]
#     dG_l = [x.dG_l for x in ps]
#     plt.plot(iterr, dG_l, 's-')
#     plt.ylabel('$\max | \Delta G_l |$')
#     plt.xlabel('Iteration')
#     plt.semilogy([], [])

#     plt.subplot(*subp); subp[-1] += 1
#     for b, g in p.G_l: p.G_l[b].data[:] = np.abs(g.data)
#     # oplotr(p.G_l['up'], 'o-', label=None)
#     oplotr(p.g_l, 'o-', label=None)
#     plt.ylabel('$| G_l |$')
#     plt.semilogy([], [])

#     plt.subplot(*subp); subp[-1] += 1
#     oplotr(p.G_tau_raw['up'], alpha=0.75, label='Binned')
#     oplotr(p.G_tau['up'], label='Legendre')
#     plt.legend(loc='best', fontsize=8)
#     plt.ylabel(r'$G(\tau)$')

#     plt.subplot(*subp); subp[-1] += 1
#     oploti(p.sigma_w[0,0], '.-', label=None)
#     plt.ylabel(r'$\Sigma(i\omega_n)$')
#     plt.xlim([0, 20])
#     plt.ylim([-1, 0])

#     plt.tight_layout()
#     # plt.savefig('figure_sc.svg')
#     plt.show()


def plot_DMFT_test(ps):
    p = ps[-1]
        
    plt.figure(figsize=(3.25*2, 5))
    subp = [2, 2, 1]

    plt.subplot(*subp); subp[-1] += 1

    iterr = [x.iter for x in ps]
    dG_l = [x.dG_l for x in ps]
    plt.plot(iterr, dG_l, 's-')
    plt.ylabel('$\max | \Delta G_l |$')
    plt.xlabel('Iteration')
    plt.semilogy([], [])

    plt.subplot(*subp); subp[-1] += 1
    # for b, g in p.g_l: p.G_l[b].data[:] = np.abs(g.data)
    p.g_l.data[:] = np.abs(p.g_l.data)
    # oplotr(p.G_l['up'], 'o-', label=None)
    oplotr(p.g_l, 'o-', label=None)
    plt.ylabel('$| G_l |$')
    plt.semilogy([], [])

    plt.subplot(*subp); subp[-1] += 1
    # oplotr(p.G_tau_raw['up'], alpha=0.75, label='Binned')
    oplotr(p.g_tau)
    # plt.legend(loc='best', fontsize=8)
    plt.ylabel(r'$G(\tau)$')

    plt.subplot(*subp); subp[-1] += 1
    oploti(p.sigma_w, '.-', label=None)
    plt.ylabel(r'$\Sigma(i\omega_n)$')
    plt.xlim([0, 20])
    plt.ylim([-1, 0])

    plt.tight_layout()
    # plt.savefig('figure_sc.svg')
    plt.show()
