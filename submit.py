import numpy as np
import sklearn

# You are allowed to import any submodules of numpy or sklearn e.g. sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_map, my_params etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	X = np.asarray( X )
	d  = 1 - 2 * X                       # {0,1} -> {+1,-1}
	de = d[:, 0::2]                       # 16 even-position signs
	do = d[:, 1::2]                       # 16 odd-position signs
	# 16 x 16 = 256 cross terms (the bilinear core of he*ho)
	cross = ( de[:, :, None] * do[:, None, :] ).reshape( X.shape[0], -1 )
	# append the 16 + 16 linear terms; the bias of LinearSVC absorbs the constant
	X_map = np.hstack( [ cross, de, do ] )   # D = 256 + 32 = 288

	return X_map

################################
# Non Editable Region Starting #
################################
def my_params( X_map, X_raw, y ):
################################
#  Non Editable Region Ending  #
################################

	my_params = {
		"loss"        : "squared_hinge",
		"C"           : 1.0,
		"tol"         : 1e-4,
		"penalty"     : "l2",
		"max_iter"    : 5000,
		"dual"        : False,
	}

	return my_params
