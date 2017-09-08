package mlia_breeze.regres.regression

import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.numerics._
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import breeze.stats._
import breeze.linalg._
import breeze.plot._
import breeze.stats.mean.reduce_Double
 
object AbaloneTest extends App {
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
  
  override def main(args: Array[String]): Unit = {
    val (dataSet, labels) = loadDataSet("regression/abalone.txt")
    val yHat01 = Regression.lwlrTest(dataSet.slice(0, 99), dataSet.slice(0, 99), labels.slice(0, 99), 0.1)
    val yHat1 = Regression.lwlrTest(dataSet.slice(0, 99), dataSet.slice(0, 99), labels.slice(0, 99), 1)
    val yHat10 = Regression.lwlrTest(dataSet.slice(0, 99), dataSet.slice(0, 99), labels.slice(0, 99), 10)
    val rss01 = Regression.rssError(yHat01.toArray, labels.slice(0, 99))
    val rss1 = Regression.rssError(yHat1.toArray, labels.slice(0, 99))
    val rss10 = Regression.rssError(yHat10.toArray, labels.slice(0, 99))
    printf("rss01: %f, rss1: %f, rss10: %f\n", rss01, rss1, rss10)

    val yHat01_t = Regression.lwlrTest(dataSet.slice(100, 199), dataSet.slice(0, 99), labels.slice(0, 99), 0.1)
    val yHat1_t = Regression.lwlrTest(dataSet.slice(100, 199), dataSet.slice(0, 99), labels.slice(0, 99), 1)
    val yHat10_t = Regression.lwlrTest(dataSet.slice(100, 199), dataSet.slice(0, 99), labels.slice(0, 99), 10)
    val rss01_t = Regression.rssError(yHat01_t.toArray, labels.slice(100, 199))
    val rss1_t = Regression.rssError(yHat1_t.toArray, labels.slice(100, 199))
    val rss10_t = Regression.rssError(yHat10_t.toArray, labels.slice(100, 199))
    printf("rss01: %f, rss1: %f, rss10: %f\n", rss01_t, rss1_t, rss10_t)

    val ws = Regression.standRegres(dataSet.slice(0, 99), labels.slice(0, 99))
    val yHat = DenseMatrix(dataSet.slice(100, 199): _*) * ws
    val rss = Regression.rssError(yHat.toArray, labels.slice(100, 199))
    printf("rss: %f\n", rss)
    
      
    val ys =  Regression.regularize(DenseMatrix(dataSet: _*)) * Regression.ridgeRegres(DenseMatrix(dataSet: _*), DenseVector(labels), 0.1)
    println(Regression.rssError(ys.toArray, labels))
    
    val wss = Regression.ridgeTest(dataSet, labels)
    println(wss)
    val rsss = (0 until wss.rows).foldLeft(DenseVector.zeros[Double](wss.rows))((rs, i) => {
      val y = Regression.regularize(DenseMatrix(dataSet: _*)) * wss(i, ::).t  + mean(labels)
      rs(i) = Regression.rssError(y.toArray, labels)
      rs
    })
    println(rsss)
    println()
    val f = Figure()
    val p = f.subplot(0)
    val x = Regression.regularize(DenseMatrix(dataSet: _*)) * wss(20, ::).t // + mean(labels)
    val y = DenseVector(labels)
   
     p += plot(yHat.toArray, labels.slice(100, 199), '.')
     
   }
}