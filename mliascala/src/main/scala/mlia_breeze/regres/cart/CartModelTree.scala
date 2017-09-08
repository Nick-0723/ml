package mlia_breeze.regres.cart

import java.io.FileOutputStream

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import mlia_breeze.regres.Common

import scala.annotation.tailrec
import mlia_breeze.regres.regression.Regression

object CartModelTree extends App {
  case class ModelTree(val feature: Int, val value: DenseVector[Double], val left: Option[ModelTree] = None, val right: Option[ModelTree] = None) {
    override def toString: String = if (!isLeaf) s"[feature: $feature, value: $value, right: ${right.getOrElse("-")}, left: ${left.getOrElse("-")}]" else s"$value"

    val isLeaf = feature == -1
    val isTree = !isLeaf

    //def value: Double = value(0)
    def modelTreeEval(inDat: DenseVector[Double]):Double = inDat dot value

    def treeForCast(inData: DenseVector[Double]): Double = {
      if (isLeaf) modelTreeEval(inData)
      else {
        if (inData(feature) > value(0)) {
          left.map(x => x.treeForCast(inData)).getOrElse(throw new IllegalStateException)
        } else {
          right.map(x => x.treeForCast(inData)).getOrElse(throw new IllegalStateException)
        }
      }
    }
 
    def createForeCast(testData: Array[Array[Double]]) = {
      val data = DenseMatrix(testData: _*)
      (0 until data.rows).foldLeft(DenseVector.ones[Double](data.rows))((yy, idx) => {
        yy(idx) = treeForCast(data(idx, ::).t)
        yy
      })
    }
  }

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

  def modelLeaf(dataSet: Array[Array[Double]]): DenseVector[Double] = linearSolve(dataSet)._1

  def modelErr(dataSet: Array[Array[Double]]): Double = {
    val (ws, x, y) = linearSolve(dataSet)
    val yHat = x * ws
    val eY = y - yHat
    sum(eY :^ 2.0)
  }

  def binSplitDataSet(dataSet: Array[Array[Double]], feature: Int, value: Double): (Array[Array[Double]], Array[Array[Double]]) = {
    @tailrec
    def split(dataSet: Array[Array[Double]], feature: Int, value: Double, i: Int, left: Array[Array[Double]], right: Array[Array[Double]]): (Array[Array[Double]], Array[Array[Double]]) = {
      if (i == dataSet.length) (left, right)
      else {
        val r = dataSet(i)
        val nLeft = if (r(feature) > value) left :+ r else left
        val nRight = if (r(feature) <= value) right :+ r else right
        split(dataSet, feature, value, i + 1, nLeft, nRight)
      }
    }
    split(dataSet, feature, value, 0, Array.empty[Array[Double]], Array.empty[Array[Double]])
  }

  def chooseBestSplit(dataSet: Array[Array[Double]], leafType: (Array[Array[Double]]) => DenseVector[Double], errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): (Option[Int], DenseVector[Double]) = {
    val Array(tolS, tolN, _*) = ops
    val data = DenseMatrix(dataSet: _*)

    if (data(::, data.cols - 1).toArray.distinct.length == 1) (None, leafType(dataSet))
    val m = data.rows
    val n = data.cols
    val S = errType(dataSet)

    val r = (0 until n - 1).foldLeft((Inf, 0, 0.0))((outer, featIndex) => {
      val features = data(::, featIndex).toArray.distinct
      features.foldLeft(outer)((inner, x) => {
        val (mat0, mat1) = binSplitDataSet(dataSet, featIndex, x)
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
      val (mat0, mat1) = binSplitDataSet(dataSet, r._2, r._3)
      if (mat0.length < tolN || mat1.length < tolN) (None, leafType(dataSet))
      else (Some(r._2), DenseVector(Array(r._3)))
    }
  }

  def createTree(dataSet: Array[Array[Double]], leafType: (Array[Array[Double]]) => DenseVector[Double], errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): ModelTree = {
    val (feat, value) = chooseBestSplit(dataSet, leafType, errType, ops)
    feat.map(x => {
      val (lSet, rSet) = binSplitDataSet(dataSet, x, value(0))
      ModelTree(x, value, Some(createTree(lSet, leafType, errType, ops)), Some(createTree(rSet, leafType, errType, ops)))
    }).getOrElse(ModelTree(-1, value, None, None))

  }
  override def main(args: Array[String]): Unit = {
     val trainData = Prep.loadDataSet("cart/bikeSpeedVsIq_train.txt")
    val modelTree = createTree(trainData, modelLeaf, modelErr, Array(1, 20))
    println(modelTree)
    val testData = Prep.loadDataSet("cart/bikeSpeedVsIq_test.txt")

    val testYHat = modelTree.createForeCast(testData)
    val data = DenseMatrix(testData: _*)
    val testY = data(::, data.cols - 1)
    val cof = Common.corrcoef(testYHat, testY)
    println(cof)
  }
}