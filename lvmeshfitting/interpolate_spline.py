# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import linalg_ops
# from tensorflow.python.ops import math_ops

EPSILON = 0.0000000001

def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0):
	r"""Interpolate signal using polyharmonic interpolation.
	The interpolant has the form
	$$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$
	This is a sum of two terms: (1) a weighted sum of radial basis function (RBF)
	terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term with a bias.
	The \\(c_i\\) vectors are 'training' points. In the code, b is absorbed into v
	by appending 1 as a final dimension to x. The coefficients w and v are
	estimated such that the interpolant exactly fits the value of the function at
	the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\), and the
	vector w sums to 0. With these constraints, the coefficients can be obtained
	by solving a linear system.
	\\(\phi\\) is an RBF, parametrized by an interpolation
	order. Using order=2 produces the well-known thin-plate spline.
	We also provide the option to perform regularized interpolation. Here, the
	interpolant is selected to trade off between the squared loss on the training
	data and a certain measure of its curvature
	([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
	Using a regularization weight greater than zero has the effect that the
	interpolant will no longer exactly fit the training data. However, it may be
	less vulnerable to overfitting, particularly for high-order interpolation.
	Note the interpolation procedure is differentiable with respect to all inputs
	besides the order parameter.
	Args:
		train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
			locations. These do not need to be regularly-spaced.
		train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional values
			evaluated at train_points.
		query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
			where we will output the interpolant's values.
		order: order of the interpolation. Common values are 1 for
			\\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\) (thin-plate spline),
			 or 3 for \\(\phi(r) = r^3\\).
		regularization_weight: weight placed on the regularization term.
			This will depend substantially on the problem, and it should always be
			tuned. For many problems, it is reasonable to use no regularization.
			If using a non-zero value, we recommend a small value like 0.001.
		name: name prefix for ops created by this function
	Returns:
		`[b, m, k]` float `Tensor` of query values. We use train_points and
		train_values to perform polyharmonic interpolation. The query values are
		the values of the interpolant evaluated at the locations specified in
		query_points.
	"""
	w, v = _solve_interpolation(train_points, train_values, order, regularization_weight)
	query_values = _apply_interpolation(query_points, train_points, w, v, order)

	return query_values

def _phi(r, order):
	"""Coordinate-wise nonlinearity used to define the order of the interpolation.
	See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
	Args:
		r: input op
		order: interpolation order
	Returns:
		phi_k evaluated coordinate-wise on r, for k = r
	"""

	# using EPSILON prevents log(0), sqrt0), etc.
	# sqrt(0) is well-defined, but its gradient is not

	if order == 1:
		# r = math_ops.maximum(r, EPSILON)
		r = torch.clamp(r, EPSILON, np.inf)
		# r = math_ops.sqrt(r)
		r = torch.sqrt(r)
		return r
	elif order == 2:
		# return 0.5 * r * math_ops.log(math_ops.maximum(r, EPSILON))
		return 0.5 * r * torch.log(torch.clamp(r, EPSILON, np.inf))
	elif order == 4:
		return 0.5 * torch.square(r) * torch.log(torch.clamp(r, EPSILON, np.inf))
	elif order % 2 == 0:
		r = torch.clamp(r, EPSILON, np.inf)
		return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
	else:
		r = torch.clamp(r, EPSILON, np.inf)
		return torch.pow(r, 0.5 * order)

def _cross_squared_distance_matrix(x, y):
	# x: [batch_size, n, d]
	# y: [batch_size, m, d]
																																											 
	x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
	y_t = y.permute(0,2,1).contiguous()
	y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
	dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
	dist[dist != dist] = 0 # replace nan values with 0

	return torch.clamp(dist, 0.0, np.inf)

def _pairwise_squared_distance_matrix(x):

	return _cross_squared_distance_matrix(x, x) #optimize later


def _solve_interpolation(train_points, train_values, order, regularization_weight):
	"""Solve for interpolation coefficients.
	Computes the coefficients of the polyharmonic interpolant for the 'training'
	data defined by (train_points, train_values) using the kernel phi.
	Args:
		train_points: `[b, n, d]` interpolation centers
		train_values: `[b, n, k]` function values
		order: order of the interpolation
		regularization_weight: weight to place on smoothness regularization term
	Returns:
		w: `[b, n, k]` weights on each interpolation center
		v: `[b, d, k]` weights on each input dimension
	"""

	b, n, d = train_points.shape
	_, _, k = train_values.shape

	# First, rename variables so that the notation (c, f, w, v, A, B, etc.)
	# follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
	# To account for python style guidelines we use
	# matrix_a for A and matrix_b for B.

	c = train_points
	f = train_values

	matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
	if regularization_weight > 0:
		batch_identity_matrix = torch.eye(n).unsqueeze(0)
		matrix_a += regularization_weight * batch_identity_matrix

	# Append ones to the feature values for the bias term in the linear model.
	ones = torch.ones([b, n, 1], device=device)
	matrix_b = torch.cat([c, ones], 2)  # [b, n, d + 1]

	# [b, n + d + 1, n]
	left_block = torch.cat([matrix_a, matrix_b.permute([0,2,1])], 1)

	num_b_cols = matrix_b.shape[2]  # d + 1
	lhs_zeros = torch.zeros([b, num_b_cols, num_b_cols], device=device)
	right_block = torch.cat([matrix_b, lhs_zeros], 1)  # [b, n + d + 1, d + 1]
	lhs = torch.cat([left_block, right_block], 2)  # [b, n + d + 1, n + d + 1]

	rhs_zeros = torch.zeros([b, d + 1, k], device=device)
	rhs = torch.cat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

	# yyr 判断lhs是否可逆
	determinant_lhs = torch.det(lhs * 1000)  
	# 处理保留到10位
	precision = 8 
	determinant_lhs = torch.trunc(determinant_lhs * 10**precision) / (10**precision)  
	if torch.eq(determinant_lhs, 0.0):
		# 求伪逆矩阵
		# print(determinant_lhs)
		lhs_pseudo_inv = torch.linalg.pinv(lhs) 
		w_v = torch.matmul(lhs_pseudo_inv, rhs)     
	else:   
		# 原来的代码，也是非奇异矩阵  
		# print(determinant_lhs)
		w_v = torch.linalg.solve(lhs,rhs)
        

	# 原来的代码
	# w_v = torch.linalg.solve(lhs,rhs)
    

	w = w_v[:, :n, :]
	v = w_v[:, n:, :]

	return w, v


def _apply_interpolation(query_points, train_points, w, v, order):
	"""
	Apply polyharmonic interpolation model to data.
	Given coefficients w and v for the interpolation model, we evaluate interpolated function values at query_points.
	Args:
		query_points: `[b, m, d]` x values to evaluate the interpolation at
		train_points: `[b, n, d]` x values that act as the interpolation centers (the c variables in the wikipedia article)
		w: `[b, n, k]` weights on each interpolation center
		v: `[b, d, k]` weights on each input dimension
		order: order of the interpolation
	Returns:
		Polyharmonic interpolation evaluated at points defined in query_points.
	"""

	batch_size = train_points.shape[0]
	num_query_points = query_points.shape[1]

	# First, compute the contribution from the rbf term.
	pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
	phi_pairwise_dists = _phi(pairwise_dists, order)

	rbf_term = torch.bmm(phi_pairwise_dists, w)

	# Then, compute the contribution from the linear term.
	# Pad query_points with ones, for the bias term in the linear model.
	query_points_pad = torch.cat([query_points, torch.ones([batch_size,num_query_points,1], device=device)], 2)
	linear_term = torch.bmm(query_points_pad, v)

	return rbf_term + linear_term





















