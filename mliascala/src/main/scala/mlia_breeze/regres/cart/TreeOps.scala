package mlia_breeze.regres.cart
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.annotation.tailrec

trait TreeOps[A] {

  def getLeaf(dataSet: DataSet): A

  def calcError(dataSet: DataSet): Double

  def doubleToValue(threshold: Double): A

  def toDouble(value: A): Double

}

trait RegOps extends TreeOps[Double] {

  def getLeaf(dataSet: DataSet): Double = mean(dataSet.rows.map(_.data.last))

  def calcError(dataSet: DataSet): Double = {
    val labels = dataSet.rows.map(_.data.last)
    val avg = mean(labels)
    labels.map(x => scala.math.pow(x - avg, 2)).sum
  }

  def doubleToValue(threshold: Double): Double = threshold

  def toDouble(value: Double): Double = value
}

trait ModelOps extends TreeOps[DenseVector[Double]] {

  def linearSolve(dataSet: DataSet): (DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    val X = DenseMatrix.ones[Double](dataSet.length, dataSet.colSize)
    val Y = dataSet.labelVec
    X(::, 1 until X.cols) := dataSet.dataMat
    val xTx = X.t * X
    if (det(xTx) == 0.0)
      throw new IllegalStateException("This matrix is singular, cannot do inverse, try increasing the second value of ops")
    val ws = inv(xTx) * (X.t * Y)
    (ws, X, Y)
  }
  
  def getLeaf(dataSet: DataSet): DenseVector[Double] = linearSolve(dataSet)._1

  def calcError(dataSet: DataSet): Double = {
    val (ws, x, y) = linearSolve(dataSet)
    val yHat = x * ws
    (y :- yHat).map(x => scala.math.pow(x, 2)).sum
  }

  def doubleToValue(threshold: Double): DenseVector[Double] = DenseVector(threshold)

  def toDouble(value: DenseVector[Double]) = value(0)
}