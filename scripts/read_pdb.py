# load just the first few frames from a giant PDB-format trajectory

import mdtraj

def extract_first_frames(filename,num_desired_frames=1):
    ''' filename is an unwieldy large pdb file, containing a large number of
    frames, separated by "ENDMDL."

    Instead of loading the entire thing into memory, we just want to load
    the first few frames, saving each as its own pdb file.

    All we do is just split the file into chunks separated at "ENDMDL."
    '''


    with open(filename) as f:
        for i in range(num_desired_frames):
            # read a frame
            line = ''
            lines = []
            while line[:6] != 'ENDMDL':
                line = f.readline()
                lines.append(line)

            # write a frame
            out_name = 'frame{0}.pdb'.format(i)
            with open(out_name,'w') as f_write:
                f_write.writelines(lines)

if __name__=='__main__':
    frames = extract_first_frames('../example-data/fs_peptide.pdb')

    # load a frame we just wrote as a Trajectory object
    frame0 = mdtraj.load('frame0.pdb')
