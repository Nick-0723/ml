package mlia_breeze.regres.cart

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.annotation.tailrec

abstract class TreeNode[T](val spInd: Int, val spVal: T, val left: Option[TreeNode[T]] = None, val right: Option[TreeNode[T]] = None) {
  override def toString: String = if (!isLeaf) s"[feature: $spInd, value: $spVal, left: ${left.getOrElse("-")}, right: ${right.getOrElse("-")}]" else s"$spVal"

  val isLeaf = spInd == -1
  val isTree = !isLeaf
  val doubleValue: Double
  
  def mean: Double
  def predict(inDat: DenseVector[Double]): Double
  def branch(left: Option[TreeNode[T]], right: Option[TreeNode[T]]): TreeNode[T]
  def leftValue: T = left.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())

  def rightValue: T = right.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())
  def prune(testData: DataSet): TreeNode[T]
  def square(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map(x => scala.math.pow(x, 2))
  def square(m: DenseVector[Double]): DenseVector[Double] = m.map(x => scala.math.pow(x, 2))

  def treeForeCast(inData: DenseVector[Double]): Double = {
    val descend: Option[TreeNode[T]] => Double = _.map(subTree => {
      if (subTree.isTree) subTree.treeForeCast(inData) else subTree.predict(inData)
    }).getOrElse[Double](throw new IllegalStateException("can not reach leaf node"))

    if (this.isLeaf) predict(inData)
    else if (inData(this.spInd) > this.doubleValue) descend(left) else descend(right)
  }

  def createForeCast(testMat: DenseMatrix[Double]) = {
    (0 until testMat.rows).foldLeft(DenseVector.zeros[Double](testMat.rows))((v, i) => {
      v(i) = treeForeCast(testMat(i, ::).t)
      v
    })
  }
}

case class RegTree(override val spInd: Int,
                   override val spVal: Double,
                   override val left: Option[TreeNode[Double]] = None,
                   override val right: Option[TreeNode[Double]] = None) extends TreeNode[Double](spInd, spVal, left, right) {

  val doubleValue: Double = spVal

  def predict(inDat: DenseVector[Double]) = doubleValue

  def branch(l: Option[TreeNode[Double]], r: Option[TreeNode[Double]]): TreeNode[Double] = this.copy(left = l, right = r)

  def mean: Double =
    (right.map(r => if (r.isTree) r.mean else spVal).getOrElse(spVal) +
      left.map(l => if (l.isTree) l.mean else spVal).getOrElse(spVal)) / 2.0

  def prune(testData: DataSet): TreeNode[Double] = {
    if (testData.length == 0) RegTree(-1, mean)
    else {
      val curTree = if (right.exists(_.isTree) || left.exists(_.isTree)) {
        val (lSet, rSet) = testData.binSplitDataSet(spInd, spVal)
        val maybeLeft = left.filter(_.isTree).map(_.prune(lSet)).orElse(left)
        val maybeRight = right.filter(_.isTree).map(_.prune(rSet)).orElse(right)
        copy(left = maybeLeft, right = maybeRight)
      } else this

      if (curTree.right.exists(_.isLeaf) && curTree.left.exists(_.isLeaf)) {
        val (lSet, rSet) = testData.binSplitDataSet(curTree.spInd, curTree.spVal)
        val errorNoMerge = sum(square(lSet.labelVec :- curTree.leftValue)) + sum(square(rSet.labelVec :- curTree.rightValue))
        val treeMean = (curTree.leftValue + curTree.rightValue) / 2.0
        val errorMerge = sum(square(testData.labelVec :- treeMean))
        if (errorMerge < errorNoMerge) {
          println("merging")
          RegTree(-1, treeMean)
        } else curTree
      } else curTree
    }
  }
}

case class ModelTree(override val spInd: Int,
                     override val spVal: DenseMatrix[Double],
                     override val left: Option[TreeNode[DenseMatrix[Double]]] = None,
                     override val right: Option[TreeNode[DenseMatrix[Double]]] = None) extends TreeNode[DenseMatrix[Double]](spInd, spVal, left, right) {

  override def toString: String = if (!isLeaf) s"[feature: $spInd, value: [${spVal.valueAt(0)}], left: ${left.getOrElse("-")}, right: ${right.getOrElse("-")}]" else s"(${spVal.valuesIterator.mkString(",")})"

  val doubleValue: Double = spVal(0, 0)

  def predict(inDat: DenseVector[Double]): Double = {
    val n = inDat.length
 //   val X = DenseMatrix.ones[Double](1, n + 1)
    val X = DenseVector.ones[Double](n + 1)
    X(1 until n + 1) := inDat
    val ss = X
    val r = X * spVal
    r(0, 0)
  }

  def branch(l: Option[TreeNode[DenseMatrix[Double]]], r: Option[TreeNode[DenseMatrix[Double]]]): TreeNode[DenseMatrix[Double]] = this.copy(left = l, right = r)

  def mean: Double = {
    import breeze.stats.{ mean => matMean }
    (right.map(r => if (r.isTree) r.mean else matMean(spVal)).getOrElse(matMean(spVal)) +
      left.map(l => if (l.isTree) l.mean else matMean(spVal)).getOrElse(matMean(spVal))) / 2.0
  }

  def prune(testData: DataSet): TreeNode[DenseMatrix[Double]] = throw new UnsupportedOperationException("ModelTree has not yet supported.")
}
