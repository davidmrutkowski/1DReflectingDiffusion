1D-ReflectingDiffusion.py seeks to determine the D, koff values based on intensity data from a linescan
from photobleached region on cell. The code was used to fit FRAP data in:

V. Gerganova, I. Lamas, D. M. Rutkowski, A. Vještica, D. G. Castro, V. Vincenzetti, D. Vavylonis, S. G. Martin, "Cell patterning by secretion-induced plasma membrane flows," (Science Advances, to appear) bioRxiv https://doi.org/10.1101/10.1101/2020.12.18.423457 (2020)

It does this by minimizing the error between a model recovery curve, 
as described in Gerganova et al. and the given recovery curve using SciPy's optimize.curve_fit function.

Input for 1D-ReflectingDiffusion.py should have position along linescan in first column.
Subsequent columns should be intensity data at these positions for each time step.
(example file given "CIBN-CRY2_231-Ch2.txt" is for an individual CIBN-RitC (+CRY2) recovery curve)

I0 parameter, intensity outside of observed region, is most important parameter to change for each cell.
This parameter can be estimated based on intensity before photobleaching along linescan.

t_step_size is the number of seconds between the snapshots of the cell (time between columns in input file).

D_guess and k_off do not have to be particularly close to the final value, but better initial guesses will
speed up curve fitting.

min_l_f and max_l_f restricts the length of region outside the observed region to realistic values 
(in Gerganova et al. min_l_f=2 um and max_l_f=10 um for most cells analyzed)
