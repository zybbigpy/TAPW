mkdir test_fig
echo 'test kd tree'
python test_kd_tree.py
echo 'test mag moire set up'
python test_mag_moire_setup.py
echo 'test frac coordinate set up' 
python test_frac_coord.py 
echo 'test mag gvector set up'
python test_mag_gevatoms.py
echo 'test tb moire set up, tb gvector set up'
python test_moire_setup.py
echo 'mag dispersion test (p=0) return to normal case'
python test_mag_disp.py 