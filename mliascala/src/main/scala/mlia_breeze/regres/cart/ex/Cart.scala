package mlia_breeze.regres.cart.ex

import breeze.linalg.{DenseMatrix, DenseVector, det, inv}
import breeze.numerics.Inf

import scala.annotation.tailrec

class Cart {
  abstract class TreeNode[T](val feature: Int, val value: T, val left: Option[TreeNode[T]] = None, val right: Option[TreeNode[T]] = None){
    val isLeaf: Boolean = feature == -1
    val isTree: Boolean = !isLeaf
    def leafType(): T = value
  }
/*
  case class ModelTree(override val feature: Int,
                       override val value: DenseVector[Double],
                       override val left: Option[TreeNode[DenseVector[Double]]],
                       override val right: Option[TreeNode[DenseVector[Double]]])
    extends TreeNode[DenseVector[Double]](feature, value, left, right){

    def linearSolve(dataSet: Array[Array[Double]]): (DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
      val data = DenseMatrix(dataSet: _*)
      val m = data.rows
      val n = data.cols
      val x = DenseMatrix.ones[Double](m, n)
      val y = DenseVector.ones[Double](m)
      x(0 until m, 1 until n) := data(0 until m, 0 until n - 1)
      y := data(::, n - 1)

      val xTx = x.t * x
      if (det(xTx) == 0.0) {
        println("this matrix is singular, cannot do inverse")
        (DenseVector.zeros[Double](0), x, y)
      } else {
        val ws = inv(xTx) * (x.t * y)
        (ws, x, y)
      }
    }

  }

  case class RegTree(override val feature: Int,
                     override val value: Double,
                     override val left: Option[TreeNode[Double]],
                     override val right: Option[TreeNode[Double]])
    extends TreeNode[Double](feature, value, left, right){
  }

  case class Row(data: Array[Double])

  object Row {

  }

  object DataSet{
    def empty: DataSet = new DataSet(Array.empty[Array[Double]])
    def apply(data: Seq[Array[Double]]): DataSet = new DataSet(data.toArray)

  }

  case class DataSet(dataSet: Array[Array[Double]]) extends Seq[Array[Double]]{
    override def iterator: Iterator[Array[Double]] = dataSet.iterator

    def binSplitDataSet(feature: Int, value: Double): (DataSet, DataSet) = {
      @tailrec
      def split(feature: Int, value: Double, i: Int, left: DataSet, right: DataSet): (DataSet, DataSet) = {
        if (i == dataSet.length) (left, right)
        else {
          val r = dataSet(i)
          val nLeft: Seq[Array[Double]] = if (r(feature) > value) left :+ r else left
          val nRight: Seq[Array[Double]] = if (r(feature) <= value) right :+ r else right
          split(feature, value, i + 1, DataSet(nLeft), DataSet(nRight))
        }
      }
      split(feature, value, 0, DataSet.empty, DataSet.empty)
    }

    def chooseBestSplit[T](leafType: (Array[Array[Double]]) => T, errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): (Option[Int], T) = {
      val Array(tolS, tolN, _*) = ops
      val data = DenseMatrix(dataSet: _*)

      if (data(::, data.cols - 1).toArray.distinct.length == 1) (None, leafType(dataSet))
      val m = data.rows
      val n = data.cols
      val S = errType(dataSet)

      val r = (0 until n - 1).foldLeft((Inf, 0, 0.0))((outer, featIndex) => {
        val features = data(::, featIndex).toArray.distinct
        features.foldLeft(outer)((inner, x) => {
          val (mat0, mat1) = binSplitDataSet(featIndex, x)
          if (mat0.length < tolN || mat1.length < tolN) inner
          else {
            val newS = errType(mat0) + errType(mat1)
            if (newS < inner._1) (newS, featIndex, x)
            else inner
          }
        })
      })
      if ((S - r._1) < tolS) (None, leafType(dataSet))
      else {
        val (mat0, mat1) = binSplitDataSet(r._2, r._3)
        if (mat0.length < tolN || mat1.length < tolN) (None, leafType(dataSet))
        else (Some(r._2), T(Array(r._3)))
      }
    }

    def createTree[T](leafType: (Array[Array[Double]]) => DenseVector[Double], errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): TreeNode[T] = {
      val (feat, value) = chooseBestSplit(leafType, errType, ops)
      feat.map(x => {
        val (lSet, rSet) = binSplitDataSet(x, value(0))
        TreeNode(x, value, Some(lSet.createTree(leafType, errType, ops)), Some(rSet.createTree(leafType, errType, ops)))
      }).getOrElse(ModelTree(-1, value, None, None))

    }

  }*/
}
