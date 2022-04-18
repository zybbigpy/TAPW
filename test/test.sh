mkdir test_fig
echo 'test kd tree'
python test_kd_tree.py
echo 'test frac coordinate set up' 
python test_frac_coord.py 
echo 'test tb moire set up, tb gvector set up'
python test_moire_setup.py
echo 'test tb moire plot'
python test_disp.py