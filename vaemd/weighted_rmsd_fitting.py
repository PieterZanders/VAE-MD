import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm

class GaussianFit():

    """
    Source:
    https://github.com/Balasubra/MolDyn_Rigidity/tree/main
    https://www.sciencedirect.com/science/article/pii/S0006349506726315#fig1
    """

    def __init__(self,Mobile,Reference,sfactor=5.0):
        self.mob , self.ref  = Mobile , Reference
        self.fac , self.sel  = sfactor , 'name CA'
        self.nframes = self.mob.trajectory.n_frames
        self.natoms  = self.mob.trajectory.n_atoms

        # Index of Ca atoms
        self.ndx   = np.where( self.mob.atoms.names == 'CA' )[0]
        self.size  = self.ndx.size

        # Get positions
        self.mpos = self.mob.atoms.positions[ self.ndx ]
        self.rpos = self.ref.atoms.positions[ self.ndx ]

        # Output data
        self.result = np.zeros( (self.nframes,2))
        self.coords = np.zeros( (self.nframes,self.natoms,3) , dtype=np.float32 )
        self.weight = np.zeros( (self.nframes,self.size))

    def g_weights(self):
        #Squared Displacement
        disp = (self.rpos - self.mpos)**2
        dsum = disp.sum(axis=1)
        return np.exp(-(dsum)/self.fac)

    def g_fits(self,frame):
        # Forward to specific frame
        self.mob.universe.trajectory[ frame ]
        wRMSD, sRMSD, Conv = 100, 0 , 0
        # Initial weight 1 for all Ca atoms
        Wgs = np.ones( self.size , dtype=np.float16 )

        # Start iterations
        for i in range(50):
            # alignto returns rot.matrix and RMSD
            _,RMSD = align.alignto(self.mob , self.ref , select=self.sel , weights=Wgs, match_atoms=True)

            # Updated positions,weights
            self.mpos = self.mob.atoms.positions[ self.ndx ]
            Wgs = self.g_weights()

            if i==0:
                # First iteration, assign RMSD to std.RMSD
                sRMSD = RMSD

            # Check convergence
            if (wRMSD - RMSD) < 0.0001:
                Conv = i+1
                self.coords[frame] = self.mob.atoms.positions
                self.weight[frame] = Wgs
                break

            wRMSD = RMSD
        self.result[frame,:] =  sRMSD,wRMSD

    def update(self,ts):
        ts.positions = self.coords[ts.frame]
        return ts

    def conclude(self,prefix='gfit',output=False):
        columns = ['sRMSD' , 'wRMSD' ]
        self.df = pd.DataFrame(self.result , columns=columns)
        self.wg = pd.DataFrame(self.weight, columns=[ f"R{i}" for i in range(1,self.size+1) ] )

        if output:
            self.df.to_csv( f'{prefix}_rms.csv',index_label='Frame')
            self.wg.to_csv( f'{prefix}_wgs.csv',index=False)
            self.mob.trajectory.add_transformations(self.update)
            sel = self.mob.select_atoms("all")
            with mda.Writer(f'{prefix}_fit.xtc', sel.n_atoms) as W:
                for ts in self.mob.trajectory:
                    W.write(sel)
            print( f"Fitted traj is writtein as {prefix}_fit.xtc")

        return self.df

    def fitted_traj(self):
        """
        Return the aligned trajectory as a numpy array of shape [frames, atoms, xyz].

        Returns:
            np.ndarray: The fitted trajectory coordinates.
        """
        return self.coords
    
def WeightedRMSDFit(pdb_path, xtc_path, sfactor, ref_path, stride):

    if not ref_path:
        ref_path = pdb_path
    
    reference = mda.Universe(ref_path)
    mobile = mda.Universe(pdb_path, xtc_path)

    # Initialize the Gaussian wRMSD alignment
    gaussian_fit = GaussianFit(Mobile=mobile, Reference=reference, sfactor=sfactor)

    # Perform alignment for all frames
    for frame in tqdm(range(0, mobile.trajectory.n_frames, stride), desc="Gaussian wRDMS Alignment"):
        gaussian_fit.g_fits(frame)
    
    return gaussian_fit.fitted_traj()