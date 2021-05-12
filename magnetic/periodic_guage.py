import math
import numpy as np


FRAC  = 1E-4
EDGE  = 1E-4
piece = [0 for i in range(100)]

def frac(frc):
    """
    Concretely, it will not provide the fractional part of a number
    """

    return frc-math.floor(frc)


def ab_phase(Ri, Rj, B):
    """
    Ri, Rj are two dimentional vectors (fraction coordinate in original 2D lattice)
    B is magnetic field.

    Periodic Landau Guage. Code Changed from Wannier Tools.
    """
    
    phase = 0
    i1 = math.floor(Ri[0]+EDGE)
    i2 = math.floor(Rj[0]+EDGE)
    k  = 0
    
    if (i1==i2):
        frazy = frac(Ri[0]+FRAC)+frac(Rj[0]+FRAC)
        phase += B*frazy*(Rj[1]-Ri[1])/2
    elif (i1<i2):
        piece[0] = Ri[0]
        k = 1
        for i in range(i1+1, i2+1):
            piece[k] = i
            k += 1
        piece[k] = Rj[0]
        kk = (Rj[1]-Ri[1])/(Rj[0]-Ri[0])
        for i in range(k):
            frazy = frac(piece[i]+FRAC)+frac(piece[i+1]-FRAC)
            z1 = (Ri[1]+kk*(piece[i]  -Ri[0]))
            z2 = (Ri[1]+kk*(piece[i+1]-Ri[0]))
            phase += B*frazy*(z2-z1)/2
            if (i>0):
                phase -= z1*B
    elif (i1>i2):
        piece[0] = Ri[0]
        k = 1 
        for i in range(i1, i2, -1):
            piece[k] = i
            k += 1
        piece[k] = Rj[0]
        kk = (Rj[1]-Ri[1])/(Rj[0]-Ri[0])
        for i in range(k):
            frazy = frac(piece[i]-FRAC)+frac(piece[i+1]+FRAC)
            z1 = (Ri[1]+kk*(piece[i]  -Ri[0]))
            z2 = (Ri[1]+kk*(piece[i+1]-Ri[0]))
            phase += B*frazy*(z2-z1)/2
            if (i>0):
                phase += z1*B
    return phase


def set_ab_phase_list(num_of_pairs, magnetic_field, atom_pstn_2darray_frac, atom_neighbour_2darray_frac):

    ab_phase_list = []
    
    for i in range(num_of_pairs):
        Ri = atom_pstn_2darray_frac[i]
        Rj = atom_neighbour_2darray_frac[i]
        ab_phase_list.append(ab_phase(Ri, Rj, magnetic_field))
    
    return np.array(ab_phase_list)



if __name__ == "__main__":
    B = -0.062831853071795798
    R1 = [0.666667, 0.33333]
    R2 = [2.666667, 0.33333]
    print(ab_phase(R1, R2, B))