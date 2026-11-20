"""Exact comparison of two preprocessing outputs (HDF files with train/test/valid_train/valid_test, or folders of them).

usage: python reference/compare_hdf.py <ref.hdf|dir> <new.hdf|dir>
Checks shape, columns, dtypes, index and every value (no tolerance). Exit 0 = identical."""
import glob, os, sys
import pandas as pd
a, b = sys.argv[1:3]
pairs = [(a, b)] if os.path.isfile(a) else [(f, os.path.join(b, os.path.basename(f))) for f in sorted(glob.glob(os.path.join(a, '*.hdf')))]
bad = 0
for fa, fb in pairs:
    for key in ['train', 'test', 'valid_train', 'valid_test']:
        da, db = pd.read_hdf(fa, key), pd.read_hdf(fb, key)
        try:
            pd.testing.assert_frame_equal(da, db, check_exact=True)
            print(f'IDENTICAL  {os.path.basename(fa)}:{key:12} {da.shape}')
        except AssertionError as e:
            bad += 1; print(f'DIFFERENT  {os.path.basename(fa)}:{key:12} {da.shape} vs {db.shape}\n   {str(e)[:300]}')
sys.exit(1 if bad else 0)
