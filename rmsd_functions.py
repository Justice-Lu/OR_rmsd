
import sys
import numpy as np


def get_coordinates_pdb(filename, is_gzip=False, return_atoms_as_int=False):
    """
    Get coordinates from the first chain in a pdb file
    and return a vectorset with all the coordinates.

    Parameters
    ----------
    filename : string
        Filename to read

    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """

    # PDB files tend to be a bit of a mess. The x, y and z coordinates
    # are supposed to be in column 31-38, 39-46 and 47-54, but this is
    # not always the case.
    # Because of this the three first columns containing a decimal is used.
    # Since the format doesn't require a space between columns, we use the
    # above column indices as a fallback.
    x_column = None
    V = list()

    # Same with atoms and atom naming.
    # The most robust way to do this is probably
    # to assume that the atomtype is given in column 3.

    atoms = list()
    resid = list()

    if is_gzip:
        openfunc = gzip.open
        openarg = "rt"
    else:
        openfunc = open
        openarg = "r"

    with openfunc(filename, openarg) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("TER") or line.startswith("END"):
                break
            if line.startswith("ATOM"):
                tokens = line.split()
                # Try to get the atomtype
                try:
                    atom = tokens[2]
                    atoms.append(atom)
                except ValueError:
                    msg = f"error: Parsing atomtype for the following line:" f" \n{line}"
                    exit(msg)

                if x_column is None:
                    try:
                        # look for x column
                        for i, x in enumerate(tokens):
                            if "." in x and "." in tokens[i + 1] and "." in tokens[i + 2]:
                                x_column = i
                                break

                    except IndexError:
                        msg = "error: Parsing coordinates " "for the following line:" f"\n{line}"
                        exit(msg)

                # Try to read the coordinates
                try:
                    V.append(np.asarray(tokens[x_column : x_column + 3], dtype=float))
                
                except ValueError:
                    # If that doesn't work, use hardcoded indices
                    try:
                        x = line[30:38]
                        y = line[38:46]
                        z = line[46:54]
                        V.append(np.asarray([x, y, z], dtype=float))
                    except ValueError:
                        msg = "error: Parsing input " "for the following line:" f"\n{line}"
                        exit(msg)
                # Try to read the resid number
                try:
                    resid.append(np.asarray(tokens[5], dtype=int))
                except ValueError:
                    msg = "error while reading resid - HL"
                    exit(msg)


#    if return_atoms_as_int:
#        atoms = [int_atom(atom) for atom in atoms]

    V = np.asarray(V)
    atoms = np.asarray(atoms)
    resid = np.asarray(resid)


    
    assert V.shape[0] == atoms.size 

    return atoms, V, resid


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)

def distance(V, W):
    """
    Calculate Root-mean-square deviation from two points V and W.

    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    return np.sqrt((diff * diff).sum())


def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U





def angle(CA, CX, Center):
    """
    Calculate angle at CA pointing towards CA-center and CA-CX. 

    Parameters
    ----------
    A : array
        [x, y, z] coordinates.
    B : array
        [x, y, z] coordinates.
    Center : array
        [x, y, z] coordinates.


    Returns
    -------
    angle : float
        Degree angle at center between A and B 
    """
    
    v1 = [CX[0] - CA[0], CX[1] - CA[1], CX[2] - CA[2]]
    v2 = [Center[0] - CA[0], Center[1] - CA[1], Center[2] - CA[2]]
    v1mag = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    v1norm = [v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag]
    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = [v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag]
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    
    angle = np.arccos(res)*(180/np.pi)
    
    return angle











