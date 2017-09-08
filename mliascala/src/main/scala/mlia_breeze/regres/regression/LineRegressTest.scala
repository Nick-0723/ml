package mlia_breeze.regres.regression

import java.io.BufferedReader
import java.io.FileReader
import java.io.File
import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.numerics._
 
object LineRegressTest extends App {
  def loadDataSet(fileName: String): (Array[Array[Double]], Array[Double]) = withIterator(fileName) { (ite, numFeat) =>
    val dataAndLabels = ite.toArray.map { line =>
      val lineArr = line.split('\t').map(_.toDouble)
      val feat = (0 until numFeat - 1).map(i => lineArr(i))
      val label = lineArr(numFeat - 1)
      (feat, label)
    }.unzip

    dataAndLabels._1.map(_.toArray).toArray -> dataAndLabels._2.toArray
  }

  private def withIterator[R](fileName: String)(f: (Iterator[String], Int) => R) = {
    val file = new File(fileName)
    val headerReader = new BufferedReader(new FileReader(file))
    val numFeat = try { headerReader.readLine().split('\t').size } finally { headerReader.close() }
    val bodyReader = new BufferedReader(new FileReader(file))
    try { f(Iterator.continually(bodyReader.readLine()).takeWhile(_ != null), numFeat) } finally { headerReader.close() }
  }
  
 
  override def main(args: Array[String]) = {
    val (dataSet, labels) = loadDataSet("regression/ex0.txt")
    println(dataSet.length)
    val ws = Regression.standRegres(dataSet, labels)
    println(ws)
    val res = DenseMatrix(dataSet: _*) * ws
    println(Regression.corrcoef(DenseVector(labels), res))
    
    val yhat = Regression.lwlrTest(dataSet, dataSet, labels, 0.001)
    println(Regression.corrcoef(DenseVector(labels), yhat))

  }
}