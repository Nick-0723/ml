package mlia_breeze.regres.cart

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import scala.annotation.tailrec
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.annotation.tailrec

import Cart._

case class Row(data: Array[Double])

case class DataSet(rows: Array[Row]) extends Seq[Row] {
  val length = rows.length
  val iterator = rows.iterator
  val colSize = if (rows.isEmpty) -1 else rows.head.data.size
  lazy val allSameLabel = rows.map(_.data.last).distinct.size == 1
  lazy val labelVec = DenseVector(rows.map(_.data.last))
  lazy val dataMat = DenseMatrix(rows.map(x => x.data.slice(0, colSize - 1)): _*)
  def toMat = DenseMatrix(rows.map(_.data): _*)
  
  def foldPredictors[R](z: R)(op: (R, Double, Int) => R): R = rows.map(_.data.slice(0, colSize - 1)).foldLeft(z)((outer, elem) => {
    elem.zipWithIndex.foldLeft(outer)((inner, vc) => op(inner, vc._1, vc._2))

  })
  
  def apply(row: Int): Row = this.rows(row)
  def apply(row: Int, col: Int): Double = rows(row).data(col)
  def cell(row: Int, col: Int): Double = apply(row, col)
  def row(row: Int): Row = rows(row)
  def binSplitDataSet(feature: Int, value: Double): (DataSet, DataSet) = {
    @tailrec
    def split(i: Int, left: DataSet, right: DataSet): (DataSet, DataSet) = {
      if (i == length) (left, right)
      else {
        val newLeft = if (cell(i, feature) > value) left :+ row(i) else left
        val newRight = if (cell(i, feature) <= value) right :+ row(i) else right
        split(i + 1, DataSet(newLeft.toArray), DataSet(newRight.toArray))
      }
    }
    split(0, DataSet.empty, DataSet.empty)
  }
  
  def createTree[T](ops: Array[Double])(implicit op: TreeOps[T] with TreeBuilder[T]): TreeNode[T] = {
    val (feat, value) = chooseBestSplit(ops)
    feat.map { idx =>
      val (lSet, rSet) = binSplitDataSet(idx, op.toDouble(value))
      op.branch(idx, value, Some(lSet.createTree(ops)), Some(rSet.createTree(ops)))
    } getOrElse op.leaf(value)
  }
  
  def chooseBestSplit[T](ops: Array[Double])(implicit model: TreeOps[T]): (Option[Int], T) = {
    val Array(tolS, tolN, _*) = ops
    // if all the target variables are the same value: quit and return value
    if (allSameLabel) (None, model.getLeaf(this))
    else {
      // the choice of the best feature is driven by Reduction in RSS error from mean
      val S = model.calcError(this)
      val finalCtx = foldPredictors(BestSplitCtx(Inf, -1, 0.0)) { 
        (curCtx, splitVal, featIndex) =>
        val (left, right) = binSplitDataSet(featIndex, splitVal)
        if (left.length < tolN || right.length < tolN) curCtx
        else {
          val newS = model.calcError(left) + model.calcError(right)
          if (newS < curCtx.bestS) BestSplitCtx(bestS = newS, bestIndex = featIndex, bestValue = splitVal) else curCtx
        }
      }
      // if the decrease (S-bestS) is less than a threshold don't do the split
      if ((S - finalCtx.bestS) < tolS) (None, model.getLeaf(this))
      else {
        val (left2, right2) = binSplitDataSet(finalCtx.bestIndex, finalCtx.bestValue)
        if (left2.length < tolN || right2.length < tolN) (None, model.getLeaf(this))
        else {
          (Some(finalCtx.bestIndex), model.doubleToValue(finalCtx.bestValue))
        }
      }
    }
  }
}

object DataSet {

  def empty: DataSet = new DataSet(Array.empty)

  def apply(arr: Array[Array[Double]]): DataSet = DataSet(arr.map(elem => Row(elem)))
}
