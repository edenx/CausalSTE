import numpy as np
import argparse
from pathlib import Path
import pickle
import ot
from .ot_maps import OptimalTransportEstimator
from .estimator import _mmd, _w2, _w1, _chisq
# Import your data generation and distance functions
from .data_utils.gen_data import *
from .data_utils.gen_rhc import *
from .data_utils.gen_lalonde import *
from .data_utils.load_meps_data import load_meps_data



def compute_dist(
	dist, d, n, r, eta, 
	verbose=False, data_path=None, synthetic_path=None, 
	**kwargs):
	# compute the distance at eta=0.25 for d=5
	# eta=0.1 for d=20

	# Generate data (using a fixed seed for consistency)
	config = {
	    'rng': np.random.default_rng(42),  # Fixed seed for precomputation
	    'n0': int(n * (1 - r)),
	    'n1': int(n * r),
	    'd': d,
	    'overlap_param': 0.0,  # Use baseline for distance computation
	}

	if dist == "norm":
		data = gen_norm_data(**config)
	elif dist == "beta":
		data = gen_beta_data(**config)
	elif dist == "rhc":
		data = load_rhc_data(
			path=data_path,
			standardize=True,
			semisynthetic=True,
			semisynthetic_path=synthetic_path
		)
		outcome_type = data['outcome_info']['type']
	elif dist == "lalonde":
		data = load_lalonde_data(
			path=data_path,
			version='psid', 
			standardize=True
		)
		outcome_type = data['outcome_info']['type']
	elif dist == "meps":
		data = load_meps_data(
			semisynthetic=True,
			semisynthetic_path=synthetic_path
		)
		outcome_type = data['outcome_info']['type']
	else:
		raise ValueError(f"Unknown dist: {dist}")
    
	print(f"  Generated data: n0={config['n0']}, n1={config['n1']}, d={d}")
	# estimate OT maps
	X_control = data['X_control']
	X_treatment = data['X_treatment']

	ot_model = OptimalTransportEstimator(method='exact', oos_method='auto')
	ot_model.estimate_inverse_transport(
	            X_control,
	            X_treatment,
	            verbose=verbose
	        )
	# compute linear interpolation
	X_trans_control = ot_model.map_treatment_to_control(X_treatment)
	X_trans = eta * X_treatment + (1 - eta) * X_trans_control

	if d <= 3:
		method = "kde"
	else:
		method = "ulsif"

	# Compute distances
	# Replace these with your actual distance computation functions

	distances = {
	    'chisq': _chisq(method, ot_model, eta, X_treatment, X_control, **kwargs),
	    'w1': _w1(X_control, X_trans),
	    'w2': _w2(X_control, X_trans),
	    'mmd': _mmd(X_control, X_trans, gamma=1.),
	    'dist': dist,
	    'd': d,
	    'n': n,
	    'r': r,
	    'eta': eta
	}

	print(f"  Computed distances:")
	print(f"    chisq: {distances['chisq']:.6f}")
	print(f"    w1: {distances['w1']:.6f}")
	print(f"    w2: {distances['w2']:.6f}")
	print(f"    mmd: {distances['mmd']:.6f}")

	return distances







def main():
	parser = argparse.ArgumentParser(
		description='Precompute distances for ATT analysis'
	)

	parser.add_argument('--dist', type=str, required=True,
						choices=['norm', 'beta', 'rhc', 'lalonde', 'meps'],
						help='Distribution type')
	parser.add_argument('--d', type=int, required=True,
						help='Dimensionality')
	parser.add_argument('--n', type=int, default=8000,
						help='Sample size')
	parser.add_argument('--r', type=float, default=0.6,
						help='Treatment proportion')
	parser.add_argument('--eta', type=float, default=1.,
						help='Linear interpolation parameter')

	parser.add_argument('--output_dir', type=str, 
						default='precomputed',
						help='Output directory for precomputed distances')
	parser.add_argument('--data_path', type=str, 
						help='Path to data file')
	parser.add_argument('--synthetic_path', type=str, 
						help='Path to semi-synthetic data file')
	args = parser.parse_args()

	# Create output directory
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("="*60)
	print("Precomputing Distances")
	print("="*60)
	print(f"Configuration:")
	print(f"  dist: {args.dist}")
	print(f"  d: {args.d}")
	print(f"  n: {args.n}")
	print(f"  r: {args.r}")
	print("="*60)

	cwd = os.getcwd()

	if args.dist in ['norm', 'beta']:
		eta=0.25
		output_file = os.path.join(cwd, output_dir, f"distances_{args.dist}_d{args.d}.pkl")
	elif args.dist == 'rhc':
		eta=0.55
		output_file = os.path.join(cwd, output_dir, f"distances_{args.dist}.pkl")
		args.data_path=os.path.join(cwd, "data/rhc_processed.csv")
		args.synthetic_path=os.path.join(cwd, "data/rhc_nightmare.csv")
	elif args.dist == "lalonde":
		eta = 0.99
		output_file = os.path.join(cwd, output_dir, f"distances_{args.dist}.pkl")
		args.data_path=os.path.join(cwd, "data/lalonde_psid.csv")
	elif args.dist == "meps":
		eta=0.7 #0.4
		output_file = os.path.join(cwd, output_dir, f"distances_{args.dist}.pkl")
		args.data_path=os.path.join(cwd, "meps_data/processed/meps_medicare_att.csv")
		args.synthetic_path=os.path.join(cwd, "meps_data/processed/meps_medicare_att_semisynthetic_severe.csv")

	# Compute distances
	distances = compute_dist(
		args.dist, args.d, args.n, args.r, eta,
		data_path=args.data_path, synthetic_path=args.synthetic_path
	)


	with open(output_file, 'wb') as f:
		pickle.dump(distances, f)

	print("="*60)
	print(f"Saved to: {output_file}")
	print("="*60)

	return 0


if __name__ == "__main__":
	import sys
	sys.exit(main())