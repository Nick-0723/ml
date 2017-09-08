package mlia_breeze.regres.regression

import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
 import breeze.stats.mean.reduce_Double
 
object StageWiseTest extends App {
  override def main(args: Array[String]) = {
    val (dataSet, labels) = Prep.loadDataSet("regression/abalone.txt")
    
    val xMat = Regression.regularize(DenseMatrix(dataSet: _*))
    val yVec = DenseVector(labels) :- mean(DenseVector(labels))
    
    val ws = Regression.stageWise(dataSet, labels, 0.001, 5000)
    val ws1 = Regression.standRegres(xMat, yVec)
    println(ws)
    println(ws1)
  }
}