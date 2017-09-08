package mlia_breeze.regres.cart

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.numerics._
import breeze.stats._
import scala.annotation.tailrec
import mlia_breeze.regres.Common

object CartRegTest extends App {
  case class TreeNode(val feature: Int, val value: Double, val left: Option[TreeNode] = None, val right: Option[TreeNode] = None) {
    override def toString: String = if (!isLeaf) s"[feature: $feature, value: $value, right: ${right.getOrElse("-")}, left: ${left.getOrElse("-")}]" else s"$value"

    val isLeaf = feature == -1
    val isTree = !isLeaf
    def branch(l: Option[TreeNode], r: Option[TreeNode]): TreeNode = this.copy(left = l, right = r)

    def mean: Double = {
      val lMean = left.map(l => if (l.isTree) l.mean else value).getOrElse(value)
      val rMean = right.map(r => if (r.isTree) r.mean else value).getOrElse(value)
      (lMean + rMean) / 2.0
    }

    def leftValue: Double = left.filter(_.isLeaf).map(_.value).getOrElse(throw new IllegalStateException())
    def rightValue: Double = right.filter(_.isLeaf).map(_.value).getOrElse(throw new IllegalStateException())

    def regTreeEval(inData: DenseVector[Double]):Double = value

    def treeForCast(inData: DenseVector[Double]): Double = {
      if (isLeaf) regTreeEval(inData)
      else {
        if (inData(feature) > value) {
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

    def prune(testData: Array[Array[Double]]): TreeNode = {
      if (testData.length == 0) TreeNode(-1, mean)
      else {
        val curTree = if (left.exists(_.isTree) || right.exists(_.isTree)) {
          val (lSet, rSet) = binSplitDataSet(testData, feature, value)
          val maybeLeft = left.filter(_.isTree).map(_.prune(lSet)).orElse(left)
          val maybeRight = right.filter(_.isTree).map(_.prune(rSet)).orElse(right)
          val nTree = copy(left = maybeLeft, right = maybeRight)
          println(nTree)
          nTree
        } else this

        val retTree = if (curTree.left.exists(_.isLeaf) && curTree.right.exists(_.isLeaf)) {
          val (lSet, rSet) = binSplitDataSet(testData, curTree.feature, curTree.value)
          val tMat = DenseMatrix(testData: _*)
          val lMat = DenseMatrix(lSet: _*)
          val rMat = DenseMatrix(rSet: _*)
          val lVec = lMat(::, lMat.cols - 1) :- curTree.leftValue
          val rVec = rMat(::, rMat.cols - 1) :- curTree.rightValue
          val lVecP = DenseVector(lVec.toArray.map(pow(_, 2)))
          val rVecP = DenseVector(rVec.toArray.map(pow(_, 2)))

          val errorNoMerge = sum(lVecP) + sum(rVecP)
          val treeMean = (curTree.leftValue + curTree.rightValue) / 2.0
          val tVec = tMat(::, tMat.cols - 1) :- treeMean
          val tVecP = DenseVector(tVec.toArray.map(pow(_, 2))) 
          val errorMerge = sum(tVecP)
          if (errorMerge < errorNoMerge) {
            println("merging")
            TreeNode(-1, treeMean)
          } else curTree
        } else curTree
         retTree
      }
    }
  }
  /*
  def binSplitDataSet(dataSet: Array[Array[Double]], feature: Int, value: Double): (Array[Array[Double]], Array[Array[Double]]) = {
    
    dataSet.foldLeft((Array.empty[Array[Double]], Array.empty[Array[Double]]))((set, x) => {
      val nLeft = if(x(feature) > value) set._1 :+ x else set._1
      val nRight = if(x(feature) <= value) set._2 :+ x else set._2
      (nLeft, nRight)
    })
  }
  * 
  */
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

  def regLeaf(dataSet: Array[Array[Double]]) = {
    val data = DenseMatrix(dataSet: _*)
    mean(data(::, data.cols - 1))
  }

  def regErr(dataSet: Array[Array[Double]]) = {
    val data = DenseMatrix(dataSet: _*)
    variance(data(::, data.cols - 1)) * data.rows
  }

  def chooseBestSplit(dataSet: Array[Array[Double]], leafType: (Array[Array[Double]]) => Double, errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): (Option[Int], Double) = {
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
      else (Some(r._2), r._3)
    }
  }

  def createTree(dataSet: Array[Array[Double]], leafType: (Array[Array[Double]]) => Double, errType: (Array[Array[Double]]) => Double, ops: Array[Int] = Array(1, 4)): TreeNode = {
    val (feat, value) = chooseBestSplit(dataSet, leafType, errType, ops)
    feat.map(x => {
      val (lSet, rSet) = binSplitDataSet(dataSet, x, value)
      TreeNode(x, value, Some(createTree(lSet, leafType, errType, ops)), Some(createTree(rSet, leafType, errType, ops)))
    }).getOrElse(TreeNode(-1, value, None, None))

  }

  override def main(args: Array[String]): Unit = {
    val trainData = Prep.loadDataSet("cart/bikeSpeedVsIq_train.txt")
    val regTree = createTree(trainData, regLeaf, regErr, Array(1, 20))
    println(regTree)
    val testData = Prep.loadDataSet("cart/bikeSpeedVsIq_test.txt")

    val testYHat = regTree.createForeCast(testData)
    val data = DenseMatrix(testData: _*)
    val testY = data(::, data.cols - 1)
    val cof = Common.corrcoef(testYHat, testY)
    println(cof)
  }
}