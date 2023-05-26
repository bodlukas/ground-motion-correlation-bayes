# -------------------
# Utilities for plotting
# -------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib as mpl
import sys
sys.path.append('../')
# -------------------
# General
# -------------------

def cm_to_inch(cm):
    return tuple([0.393701*i for i in cm])
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 9
def updaterc():

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "cm",
        }
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    plt.rcParams.update(rc)

# -------------------
# Figure 1
# -------------------
from models_numpy import rhoE, rhoEA, rhoEAS
def fig1_a_to_d(fig, grid_spec, psi_E, psi_A, psi_S):
    cmap = plt.get_cmap("Blues")
    updaterc()
    ax1 = fig.add_subplot(grid_spec[0],projection='polar')
    r = np.array([3, 4])
    thetas = np.array([10, 65])*np.pi/180
    ax1.scatter(thetas, r, marker='o', color='none', 
        s=20, lw=1.5, edgecolor='black')
    ax1.scatter(np.pi/4, 0.1, marker='*', color='none', 
        s=250, zorder=3, edgecolor='black')
    ax1.plot(np.ones(2)*thetas[0],np.array([0, r[0]]), 
        color='black' , lw=1, ls='--')
    ax1.plot(np.ones(2)*thetas[1],np.array([0, r[1]]), 
        color='black' , lw=1, ls='--')
    ax1.plot(np.linspace(thetas.min(),thetas.max(),100), 
        np.ones(100), color='tab:red', lw=1.5)
    ax1.plot(thetas, r, color='tab:red', lw=1.5)
    ax1.set_rmax(4.5); ax1.grid(color='white')
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_xlim([0, np.pi/2]); ax1.set_rticks([1,2,3,4])
    ax1.set_rlabel_position(0*np.pi/180)
    ax1.set_ylabel('Epicentral distance [km]')    
    ax1.text(0.85, 0.85, r'$\theta \ [\degree]$', 
        transform=ax1.transAxes, fontsize=MEDIUM_SIZE)
    ax1.text(30/180*np.pi, 1.3, r'$d_{\mathrm{A}}$', 
        fontsize=8, color='tab:red')
    ax1.text(40/180*np.pi, 3.2, '$d_{\mathrm{E}}$', 
        fontsize=8, color='tab:red')
    ax1.text(0.22, -0.025, r'$d_{\mathrm{S}}=\mid v_{s30,i}-v_{s30,j} \mid$', 
        fontsize=8, color='tab:red', ha='left', va='bottom',transform=ax1.transAxes)
    ax1.text(-0.12, -0.13, r'epicenter', 
        fontsize=8, color='black', ha='left', va='bottom',transform=ax1.transAxes)
    ax1.text(0.01, 0.725, r'site $i$',
        fontsize=8, ha='left', va='bottom',transform=ax1.transAxes)
    ax1.text(0.7, 0.35, r'site $j$',
        fontsize=8, ha='left', va='top',transform=ax1.transAxes)


    ax2 = fig.add_subplot(grid_spec[1])
    x = np.linspace(0, 40, 1000)
    colors = cmap([1.0, 0.6])
    lsolid = []; ldashed = []
    for le,col in zip(psi_E['LE'], colors):
        lsolid.append(ax2.plot(x, np.exp(-np.power(x/le, psi_E['gammaE'][0])), color=col, lw=1,
        label=str(le)))
        ldashed.append(ax2.plot(x, np.exp(-np.power(x/le, psi_E['gammaE'][1])), color=col, 
        ls='--', lw=1))
    ax2.set_xlabel('Euclidean distance $d_{\mathrm{E}}$ [km]')
    ax2.set_ylabel('Correlation')
    legend = ax2.legend(title='$\ell_{\mathrm{E}}$ [km]')
    plt.gca().add_artist(legend)
    plt.setp(legend.get_title(),fontsize=MEDIUM_SIZE)
    legend2 = ax2.legend([lsolid[0][0], ldashed[0][0]], 
        ['1.0', '0.5'],  title='$\gamma_{\mathrm{E}}$ [-]', loc='upper center')
    plt.setp(legend2.get_title(),fontsize=MEDIUM_SIZE)

    ax3 = fig.add_subplot(grid_spec[2])
    x = np.linspace(0, 180, 1000)
    colors = cmap([1.0, 0.8, 0.6, 0.4])
    for col, la in zip(colors, psi_A['LA']):
        ax3.plot(x, (1 + x/la)*np.power(1-x/180, 180/la), 
            color=col, lw=1, label=str(la))
    ax3.set_xlabel(r'Angular distance $d_{\mathrm{A}}$ [°]')
    ax3.legend()
    ax3.set_xticks([0, 30, 60, 90, 120, 150, 180])
    legend = ax3.legend(title='$\ell_{\mathrm{A}}$ [°]')
    plt.setp(legend.get_title(),fontsize=MEDIUM_SIZE)
    ax3.set_yticklabels([])

    ax4 = fig.add_subplot(grid_spec[3])
    x = np.linspace(0,580,1000)
    for ls,col in zip(psi_S['LS'],  colors):
        ax4.plot(x, np.exp(-np.power(x/ls, 1)), color=col, lw=1,
        label=str(ls))
    ax4.set_xlabel('Soil dissimilarity $d_{\mathrm{S}}$ [m/s]')
    legend = ax4.legend(title='$\ell_{\mathrm{S}}$ [m/s]')
    plt.setp(legend.get_title(),fontsize=MEDIUM_SIZE)
    ax4.set_yticklabels([])

def fig1_e_to_h(fig, grid_spec, df, ref_idx, kE, kEA, kEAS):
    # Store input matrix X

    grid_shape = (int(np.sqrt(len(df))), int(np.sqrt(len(df))))
    Xcgrid = df['x'].values.reshape(grid_shape)
    Ycgrid = df['y'].values.reshape(grid_shape)
    vals = [df.vs30.values.reshape(grid_shape),
        kE.reshape(grid_shape), kEA.reshape(grid_shape), 
        kEAS.reshape(grid_shape)]
    lev = np.arange(0,1.2,0.2)

    x_ref, y_ref = (df.x[ref_idx], df.y[ref_idx])
    x_event, y_event = (-5, 10)
    for i, val in enumerate(vals):
        ax = fig.add_subplot(grid_spec[0,i])
        if i == 0:
            im = ax.pcolormesh(Xcgrid, Ycgrid, val, 
                vmin=100, vmax=800, cmap=plt.get_cmap('Greens'), 
                shading='gouraud',alpha=1)
        else:
            im = ax.contourf(Xcgrid, Ycgrid, val, 
                vmin=0, vmax=1, cmap=plt.get_cmap('Blues'),
                levels=lev, alpha=0.8)
        ax.scatter(x_ref, y_ref, marker='x', 
            color='black', s=30, alpha=1, lw=1.5)
        ax.scatter(x_event, y_event, marker='*', edgecolor='black',
            color='none', s=50, lw=1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting [km]')
        if i==0: ax.set_ylabel('Northing [km]')
        if i==1:
            ax.text(-5, 6.5, 'epicenter', fontsize=8, ha='center')
            ax.annotate("reference site",
                        xy=(x_ref, y_ref-0.5), xycoords='data',
                        xytext=(x_ref, -11.3), textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3", lw=0.8),
                        fontsize=8, ha='center',va='top')
    
    cax = fig.add_subplot(grid_spec[1,0])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Greens'), 
        norm=plt.Normalize(vmin=100, vmax=800))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, label='$v_{s30}$ [m/s]', 
        orientation='horizontal', alpha=1.0)
    cax = fig.add_subplot(grid_spec[1,2])
    cbar = fig.colorbar(im, cax=cax, label='Correlation', 
        orientation='horizontal', alpha=1.0)

# -------------------
# Figure 2
# -------------------
from models_numpy import getEucDistance, getAngDistance, getSoilDissimilarity

# Helper functions to count station pairs
def get_npairs_vs30eucl(X, bin_s, bin_e):
    dist_e = getEucDistance(X)
    dist_s = getSoilDissimilarity(X)
    nPairs = np.zeros((len(bin_e), len(bin_s)))
    for i, be in enumerate(bin_e):
        for j, bs in enumerate(bin_s):
            nP = np.sum( (dist_e > be[0]) & (dist_e <= be[1]) 
                & (dist_s > bs[0]) & (dist_s <= bs[1])) /2
            nPairs[i, j] = int(nP)
    return nPairs

def get_npairs_angeucl(X, bin_a, bin_e):
    dist_e = getEucDistance(X)
    dist_a = getAngDistance(X)
    nPairs = np.zeros((len(bin_e), len(bin_a)))
    for i, be in enumerate(bin_e):
        for j, ba in enumerate(bin_a):
            ba0 = ba[0]/180; ba1 = ba[1]/180
            nP = np.sum( (dist_e > be[0]) & (dist_e <= be[1]) 
                & (dist_a > ba0) & (dist_a <= ba1)) /2
            nPairs[i, j] = int(nP)
    return nPairs

def fig2(fig, axs, bin_e, bin_a, bin_s, nP1, nP2):
    # Compute bin means for plotting
    mbs = np.stack([np.mean(k) for k in bin_s] )
    mbe = np.stack([np.mean(k) for k in bin_e] )
    mba = np.stack([np.mean(k) for k in bin_a] )
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0,1,8))
    cmap = mpl.colors.ListedColormap(colors)
    ax = axs[0]
    Mba, Mbe = np.meshgrid(mba,mbe)
    im = ax.pcolormesh(Mbe, Mba, nP2, shading='nearest', alpha=0.75, 
        cmap = cmap, norm=mpl.colors.LogNorm(vmin=1, vmax=10000))
    im.set_linewidth(0.001)
    ax.set_yticks(np.arange(0,200,20))
    ax.set_xticks(np.arange(0,45,5))
    ax.set_ylabel('Angular distance $d_{\mathrm{A}}$ [°]')
    ax.set_xlabel('Euclidean distance $d_{\mathrm{E}}$ [km]')
    ax = axs[1]
    Mbs, Mbe = np.meshgrid(mbs,mbe)
    im = ax.pcolormesh(Mbe, Mbs, nP1, shading='nearest', alpha=0.75,
        cmap = cmap, norm=mpl.colors.LogNorm(vmin=1, vmax=10000))
    ax.set_yticks(np.arange(0,550,50))
    ax.set_xticks(np.arange(0,45,5))
    ax.set_ylabel('Soil dissimilarity $d_{\mathrm{S}}$ [m/s]')
    ax.set_xlabel('Euclidean distance $d_{\mathrm{E}}$ [km]')
    cax = axs[2]
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", 
        ticks=[3, 10, 30, 100, 300, 1000, 3000],
        label='Number of station pairs', drawedges=True)
    # cbar.outline.set_linewidth(0.4)
    # cbar.outline.set_color('black')
    # cbar.dividers.set_linewidth(0.4)
    # cbar.dividers.set_color('black')
    cbar.ax.set_yticklabels(['3', '10','30','100','300','1000', '3000'])
    # cax.grid(True, linewidth=0.4)

# -------------------
# Figure 3
# -------------------
from scipy.stats import gaussian_kde
def fig3(axss, dfs, titles):
    axs = axss[0,:]
    for i, ax in enumerate(axs):
        df_es_single = dfs[i]
        x = df_es_single['LE'].values
        y = df_es_single['gammaE'].values
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k', zorder=3, linewidths=0.8)
        lev = cset.levels
        lev = lev[1:]
        cfset = ax.contourf(xx, yy, f, lev, cmap='Blues', extend='neither', 
            alpha=1, zorder=2)
        ax.scatter(x, y, color='black', alpha=0.5, s=0.3, zorder=4, marker='.', lw=0.8)
        if i==0: ax.set_ylabel(r'Exponent $\gamma_{\mathrm{E}}$ [-]')
        ax.set_xlabel(r'Lengthscale $\ell_{\mathrm{E}}$ [km]')
        ax.set_xlim([4, 46])
        ax.set_ylim([0.26, 1.24])
        ax.set_title(titles[0][i], fontsize=9)
        ax.text(0.5, 0.98, titles[1][i], fontsize=9, transform=ax.transAxes, ha='center', va='top')
        if i==1:
            ax.text(0.68, 0.3, r'sample $\mathbf{\psi}_{\mathrm{E},r}$', 
                transform=ax.transAxes, 
                fontsize=9)
            ax.annotate("kernel density",
                    xy=(12, 0.6), xycoords='data',
                    xytext=(25, 0.32), textcoords='data',
                    arrowprops=dict(arrowstyle="->",lw=0.8, 
                                    connectionstyle="angle,angleA=-180,angleB=90,rad=0"),
                    fontsize=9)
            ax.scatter(31.45, 0.572, marker='o', color='none', edgecolor='black', lw=1, s=30)
    axs = axss[1,:]
    for k, ax in enumerate(axs):
        df_es_single = dfs[k]
        for i, row in df_es_single.iterrows():
            h = np.linspace(0,40,500)
            ax.plot(h, np.exp(-np.power(h/row.LE, row.gammaE)), 
                color='tab:blue', lw=0.05, alpha=0.25)
        ax.grid(True)
        ax.set_ylim([0,1])
        ax.set_xlim([0,40])
        if k==0: ax.set_ylabel('Correlation')
        ax.set_xlabel('Euclidean distance $d_{\mathrm{E}}$ [km]')  

# -------------------
# Figure 4
# -------------------
def rhoEdist(dE, psi):
    corr = (np.exp(-np.power(dE/psi.LE, psi.gammaE)))
    return corr

def rhoEASdist(dE, dA, dS, psi):
    corr = (np.exp(-np.power(dE/psi.LE, psi.gammaE)) * 
                (psi.w * (1 + dA/psi.LA) * np.power(1 - dA/180, 180/psi.LA) + 
                (1-psi.w) * np.exp(-dS/psi.LS)))
    return corr
def fig4(axs, dA, dS, dfE, dfEAS):
    cmap = plt.get_cmap('Oranges')
    colors = cmap([1.0, 0.7, 0.5])
    dE = np.linspace(0,40,1000)
    for i,ax in enumerate(axs):
        for j in range(len(dS)):
            rho = []
            for _, psi_r in dfEAS.iterrows():
                rho.append(rhoEASdist(dE, dA[i], dS[j], psi_r))
            rho = np.vstack(rho)
            qs = np.quantile(rho, [0.05, 0.95], axis=0)
            me = rhoEASdist(dE, dA[i], dS[j], dfEAS.mean())
            ax.plot(dE, me, color=colors[j], lw=1.5, zorder=5,
                label='$d_{\mathrm{S}}=' + str(dS[j])+ '\ \mathrm{m/s}$')
            ax.fill_between(dE, qs[1,:], qs[0,:], color=colors[j], alpha=0.4, zorder=4)
        rho = []
        for _, psi_r in dfE.iterrows():
            rho.append(rhoEdist(dE, psi_r))
        rho = np.vstack(rho)
        qs = np.quantile(rho, [0.05, 0.95], axis=0)
        me = rhoEdist(dE, dfE.mean())    
        ax.plot(dE, me, color='tab:purple', lw=1.5, zorder=3)
        ax.fill_between(dE, qs[1,:], qs[0,:], color='tab:purple', alpha=0.4, zorder=2)
        if i == 1: ax.text(5, 0.6, 'Model E', color='tab:purple', 
            fontsize=8, fontweight='bold')
        ax.set_ylim([0,1])
        ax.grid(True, 'both')
        ax.set_xlim([0, 40])
        # ax.legend(title='Soil diss. $d_s$ [m/s]')
        legend = ax.legend(title='Model EAS')
        plt.setp(legend.get_title(),fontsize=8, fontweight='bold')
        ax.set_title('$d_{\mathrm{A}} = ' + str(int(dA[i])) + '\degree$', fontsize=9)
        ax.set_xlabel('Euclidean distance $d_{\mathrm{E}}$ [km]')
        if i==0: ax.set_ylabel('Correlation')

# -------------------
# Figure 5
# -------------------
def fig5(ax, vals, lppd):
    low_lim = np.floor(np.array([val.min() for val in vals]).min())
    up_lim = np.ceil(np.array([val.max() for val in vals]).max())
    bins = np.linspace(low_lim, up_lim, 70)
    colors = ['tab:blue', 'tab:purple', 'tab:orange']
    factors = [1.5, 1.05, 1.15]
    texts = [('Model E', 'event-specific'), 
        ('Model E', 'pooled'),
        ('Model EAS', 'pooled')]

    c = 0
    for i, val in enumerate(vals):
        ax.hist(val, bins, 
            alpha=0.7, edgecolor='black', color=colors[i], 
            weights=np.ones_like(val)/(val.size)*factors[i], 
            bottom = c,
            zorder=4-c, lw=0.8)
        ax.axhline(c, color='black', zorder=4-c, lw=0.8)
        ax.scatter(lppd[i], c, color='black', 
                lw=1.5, s=50, marker='x', zorder=5)
        ax.text(-152.2, c + 0.08, texts[i][0],  
        fontsize=8, ha='left', va='bottom', fontweight='bold')
        ax.text(-152.2, c + 0.02, texts[i][1],  
            fontsize=8, ha='left', va='bottom')
        c += 0.25
    ax.set_ylim([-0.1,0.80])
    ax.set_xlim(left=-152.4)
    ax.set_xlabel('Log Posterior Predictive Density (LPPD)')
    ax.set_yticks([])
    ax.set_ylabel('Number of samples')

# -------------------
# Figure 6
# -------------------
def fig6(axs, vals, num_recs, mags):
    xaxis = num_recs
    mag_mp = [3.5, 4.5, 5.5, 6.5, 7.5]
    bw = 1
    mags_d = []
    for mag in mags:
        md = (mag_mp[0] * (mag<=(mag_mp[0]+bw/2)) + 
            mag_mp[1] * (mag>(mag_mp[1]-bw/2)) * (mag<=(mag_mp[1]+bw/2)) + 
            mag_mp[2] * (mag>(mag_mp[2]-bw/2)) * (mag<=(mag_mp[2]+bw/2)) +
            mag_mp[3] * (mag>(mag_mp[3]-bw/2)) * (mag<=(mag_mp[3]+bw/2)) +
            mag_mp[4] * (mag>(mag_mp[4]-bw/2)) )
        mags_d.append(md)


    cmap = plt.get_cmap('inferno_r')
    normc = mpl.colors.BoundaryNorm(np.arange(3, 9, 1), cmap.N)

    texts = [('Model E', 'pooled'),
        ('Model EAS', 'pooled')]

    for i, ax in enumerate(axs):
        ax.axhline(0, color='black', ls=':', zorder=1, lw=1)
        im = ax.scatter(xaxis, vals[i], 
            c=mags_d, alpha=0.7, marker='o', zorder=2, cmap=cmap, norm=normc,
            edgecolor='black', s=15, linewidths=0.8)
        ax.set_ylim([-7, 7])
        ax.set_xlim(left=38, right=650)
        ax.set_xscale('log')
        ax.set_xlabel('Number of records')
        if i == 0: ax.set_ylabel('Rel. difference in LPPD [%]')
        ax.set_xticks([40, 60, 100, 400, 600])
        ax.set_xticklabels(['40', '60', '100', '400', '600'])
        ax.grid(True, 'both')
        ax.text(600, -5, texts[i][0],  
            fontsize=8, ha='right', va='bottom', fontweight='bold')
        ax.text(600, -6, texts[i][1],  
            fontsize=8, ha='right', va='bottom')
    cbar = plt.colorbar(im, label='Magnitude', drawedges=True)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_color('black')
    cbar.dividers.set_linewidth(0.4)
    cbar.dividers.set_color('black')

# -------------------
# Figure 7
# -------------------

def fig7(axs, valsA, valsB, models, periods):
    cmap = plt.get_cmap('Blues')
    colors = cmap([0.3, 0.7, 1.0])
    markers = ['^', 's', 'o']
    ax = axs[0]
    for i in [0, 1, 2]:
        ax.plot(periods, valsA[i], color=colors[i], marker=markers[i],
            ls='--', markersize=4, markeredgecolor='black', alpha=0.75, 
            lw=0.8, label='Model ' + models[i])
        if i == 0: low = 0.0
        else: low = valsA[i-1]
        ax.fill_between(periods, low, valsA[i], 
            alpha=0.5, color=colors[i])
    ax.axhline(0, color='black', lw=0.8)
    ax.vlines(periods, 0, valsA[-1], color='black', lw=0.2, 
        zorder=1)
    ax.set_xscale('log')
    ax.set_ylim(bottom=-0.5)
    ax.set_xticks(periods)
    ax.set_xticklabels([str(i) for i in periods])
    ax.set_ylabel('Rel. difference in LPPD [%]')
    ax.set_xlabel('Period T [s]')
    ax.legend(loc='upper left')
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none')    
    ax.text(0.5, 0.05, 'Baseline: LPPD of independent model', transform=ax.transAxes, 
        fontsize=8, ha='center', va='bottom', bbox=props)
    ax = axs[1]
    for i in np.arange(1,len(models)):
        ax.plot(periods, valsB[i], color=colors[i], marker=markers[i], ls='--', 
            markersize=4, markeredgecolor='black', alpha=0.75, 
            lw=0.8, label='Model ' + models[i])
        if i == 1: low = 0.0
        else: low = valsB[i-1]
        ax.fill_between(periods, low, valsB[i], 
            alpha=0.5, color=colors[i]) 
    ax.axhline(0, color='black', lw=0.8)
    ax.vlines(periods, 0, valsB[-1], color='black', lw=0.2, 
        zorder=1)
    ax.set_xscale('log')
    ax.set_ylim(top=3.2, bottom=-0.07)
    ax.set_xticks(periods)
    ax.set_xticklabels([str(i) for i in periods])
    ax.set_xlabel('Period T [s]')
    ax.legend(loc='upper left')
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none')    
    ax.text(0.5, 0.05, 'Baseline: LPPD of model E', transform=ax.transAxes, 
        fontsize=8, ha='center', va='bottom', bbox=props)

# -------------------
# Figure 8 (bottom row)
# -------------------
def fig8(ax, vals, lppd, models, k):
    low_lim = np.floor(np.array([val.min() for val in vals]).min())
    up_lim = np.ceil(np.array([val.max() for val in vals]).max())
    bins = np.linspace(low_lim, up_lim, 50)
    colors = ['tab:purple', 'tab:orange']

    for i, val in enumerate(vals):
        ax.hist(val, bins, 
            alpha=0.7, edgecolor='black', color=colors[i], 
            weights=np.ones_like(val)/(val.size), 
            bottom = 0,
            zorder=4, lw=0.8, label='Model ' + models[i])
        ax.axhline(0, color='black', zorder=5, lw=0.8)
        ax.scatter(lppd[i], 0, color='black', 
                lw=1.5, s=50, marker='x', zorder=5)
    ylim = ax.get_ylim()
    ax.set_ylim(bottom = -0.1*ylim[1])
    ax.set_xlabel('Rel. difference in LPPD [%]')
    if k== 0: 
        ax.legend(loc='upper right')
        ax.set_ylabel('Frequency')

# -------------------
# Figure 10
# -------------------
def ecdf(sim, x):
    F = []
    for i in x:
        F.append(1/len(sim) * np.sum(sim<=i, axis=0))
    return np.stack(F)

def fig10(ax, vals, xaxis):
    props = dict(boxstyle='circle', facecolor='white', alpha=1)
    ax.plot(xaxis, vals[0],
        color='tab:red', lw=1.75, label='Posterior predictive', ls='-', zorder=1)
    ax.plot(xaxis, vals[1],
        color='black', lw=1.5, label='Mean posterior parameters', ls='--', zorder=2)
    ax.set_yscale('log')
    ax.set_ylim([0.01, 1])
    ax.grid(axis='y', which='both')
    legend = ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('$P(A_{\{Sa(1s)>sa\}}>a|rup)$')
    ax.set_xlabel('Proportion of sites $a$')    
    ax.text(0.3, 0.09, '2', transform=ax.transAxes, fontsize=8,
        va='center', ha='center', bbox=props)
    ax.text(0.24, 0.09, 'Subregion', transform=ax.transAxes, fontsize=8,
        va='center', ha='right')   

# -------------------
# Figure 11
# -------------------
def fig11(axss, dfs, layout='gray'):
    xaxis = dfs[0]['a'].values
    props = dict(boxstyle='circle', 
        facecolor='white', alpha=1)
    for i, col in enumerate(dfs[0].columns.values[1:]):
        ax = list(axss.flatten())[i]
        ax.plot(xaxis, dfs[0][col], color='black', lw=1.5, 
                label='E', ls='--', zorder=2)
        ax.plot(xaxis, dfs[1][col], color='tab:orange', lw=1.75, 
                    label='EAS', ls='-', zorder=1)
        ax.set_yscale('log')
        ax.set_ylim([0.01, 1])
        if i == 0:
            legend = ax.legend(title='Model', loc='lower left')
            plt.setp(legend.get_title(),fontsize=8, fontweight='bold')
        if (i == 0) | (i==3): 
            ax.set_ylabel('$P(A_{\{Sa>sa\}}>a|rup)$')
        if i>2: ax.set_xlabel('Proportion of sites $a$')
        if layout=='gray':
            ax.grid(axis='y',which='both')
        else:
            ax.grid(axis='y',which='both', color='gray', lw=0.2)
            ax.grid(axis='x', color='gray', lw=0.2)
        if len(col.split('_')) == 1:
            nsr = col[-1]
            ax.text(0.6, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.53, 0.9, 'Subregion', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='right')
        else:
            col_t1, col_t2 = col.split('_')
            nsr = col_t1[-1]
            ax.text(0.6, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.68, 0.9, '&', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center')
            nsr = col_t2[-1]
            ax.text(0.764, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.53, 0.9, 'Subregions', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='right')

# -------------------
# For supplementary material (comparison across periods)
# -------------------
def fig_Supp_CS(axss, dfs, layout='gray'):
    xaxis = dfs[0]['a'].values
    props = dict(boxstyle='circle', 
        facecolor='white', alpha=1)
    for i, col in enumerate(dfs[0].columns.values[1:]):
        ax = list(axss.flatten())[i]
        ax.plot(xaxis, dfs[0][col], color='tab:green', lw=1.5, 
                label='T = 0.3s', ls='-', zorder=1)
        ax.plot(xaxis, dfs[1][col], color='tab:orange', lw=1.75, 
                    label='T = 1s', ls='-', zorder=2)
        ax.plot(xaxis, dfs[2][col], color='black', lw=1.75, 
                    label='T = 3s', ls='-', zorder=1)
        ax.set_yscale('log')
        ax.set_ylim([0.01, 1])
        if (i == 0) | (i==3): 
            ax.set_ylabel('$P(A_{\{Sa>sa\}}>a|rup)$')
        if i == 0:
            legend = ax.legend(title='Sa(T)', loc='lower left')
            plt.setp(legend.get_title(),fontsize=8, fontweight='bold')
        if i>2: ax.set_xlabel('Proportion of sites $a$')
        if layout=='gray':
            ax.grid(axis='y',which='both')
        else:
            ax.grid(axis='y',which='both', color='gray', lw=0.2)
            ax.grid(axis='x', color='gray', lw=0.2)
        if len(col.split('_')) == 1:
            nsr = col[-1]
            ax.text(0.6, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.53, 0.9, 'Subregion', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='right')
        else:
            col_t1, col_t2 = col.split('_')
            nsr = col_t1[-1]
            ax.text(0.6, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.68, 0.9, '&', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center')
            nsr = col_t2[-1]
            ax.text(0.764, 0.9, nsr, transform=ax.transAxes, fontsize=8.5,
                va='center', ha='center', bbox=props)
            ax.text(0.53, 0.9, 'Subregions', transform=ax.transAxes, fontsize=8.5,
                va='center', ha='right')