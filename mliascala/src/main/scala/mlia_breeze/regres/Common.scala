package mlia_breeze.regres

import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics._
import breeze.stats._

object Common {
   def corrcoef(x1: DenseVector[Double], x2: DenseVector[Double]) = {
    val cr = DenseMatrix.zeros[Double](2, 2)
    cr(0, 0) = corrcoef1(x1, x1)
    cr(0, 1) = corrcoef1(x1, x2)
    cr(1, 0) = corrcoef1(x2, x1)
    cr(1, 1) = corrcoef1(x2, x2)
    cr
  }
  def corrcoef1(x1: DenseVector[Double], x2: DenseVector[Double]) = {
    val x3 = x1 :+= -mean(x1)
    val x4 = x2 :+= -mean(x2)

    (x3 dot x4) / sqrt(sum(x3 * x3) * sum(x4 * x4))
  }
    def rssError(yArr: Array[Double], yHatArr: Array[Double]): Double =
     rssError(DenseVector(yArr), DenseVector(yHatArr))
     
    def rssError(yArr: DenseVector[Double], yHatArr: DenseVector[Double]) = {
      sum((yArr - yHatArr) :^ 2.0)  
    }
}