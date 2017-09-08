package mlia_breeze.regres.cart

import breeze.linalg.DenseMatrix

trait TreeBuilder[T] {
  def leaf(splitVal: T): TreeNode[T]

  def branch(feature: Int, splitValue: T, left: Option[TreeNode[T]], right: Option[TreeNode[T]]): TreeNode[T]
}

trait RegTreeBuilder extends TreeBuilder[Double] {

  def leaf(splitVal: Double): TreeNode[Double] = this.branch(-1, splitVal, None, None)

  def branch(feature: Int, splitValue: Double, left: Option[TreeNode[Double]], right: Option[TreeNode[Double]]): TreeNode[Double] = RegTree(feature, splitValue, left, right)
}

trait ModelTreeBuilder extends TreeBuilder[DenseMatrix[Double]] {

  def leaf(splitVal: DenseMatrix[Double]): TreeNode[DenseMatrix[Double]] = this.branch(-1, splitVal, None, None)

  def branch(feature: Int, splitValue: DenseMatrix[Double], left: Option[TreeNode[DenseMatrix[Double]]], right: Option[TreeNode[DenseMatrix[Double]]]): TreeNode[DenseMatrix[Double]] = ModelTree(feature, splitValue, left, right)
}