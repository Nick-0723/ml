package mlia_breeze.regres.cart

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.annotation.tailrec

object Cart {
  case class BestSplitCtx[Leaf](bestS: Leaf, bestIndex: Int = 0, bestValue: Leaf)

  object Regression extends RegOps with RegTreeBuilder

  object Model extends ModelOps with ModelTreeBuilder
  /**
   * Calculate covariance value.
   */
  def cov(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = 
    ((x :- mean(x): DenseMatrix[Double]) :* (y :- mean(y): DenseMatrix[Double]): DenseMatrix[Double]).sum / (x.rows - 1)

  /**
   * Calculate Pearson correlation coefficient.
   */
  def cor(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = cov(x, y) / (stddev(x) * stddev(y))
 
  

}