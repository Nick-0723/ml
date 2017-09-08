package mlia_breeze.classifier.lr

import scala.io.Source
import breeze.linalg.*
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

object LRTest extends App {
  def loadDataSet(fileName: String): (List[Array[Double]], Array[Int]) = {
    Source.fromFile(fileName).getLines().toArray.foldLeft((List.empty[Array[Double]], Array.emptyIntArray))((res, line) => {
      val tokens = line.split("\\t")
      val array = Array(1.0, tokens(0).toDouble, tokens(1).toDouble)
      val t1 = res._1 :+ array
      val t2 = res._2 ++ Array(tokens(2).toInt)
      (t1, t2)
    })
  }
  
  override def main(args: Array[String]): Unit = {
    val (dataSet, labels) = loadDataSet("lr/testSet.txt")
    val error1 = dataSet.zipWithIndex.foldLeft((LogisticRegression.gradAscent(dataSet, labels), 0))((curr, row) => {
      val sss = curr._1
      val weight: DenseMatrix[Double] = curr._1.toDenseMatrix
      val res = LogisticRegression.classifyVector(DenseVector(row._1), weight(::, 0))
      val err = if (labels(row._2) != res) curr._2 + 1
      else curr._2
      (curr._1, err)
    })._2
    println(error1.toDouble / dataSet.length)
    println()
    val error2 = dataSet.zipWithIndex.foldLeft((LogisticRegression.stocGradAscent0(dataSet, labels), 0))((curr, row) => {
      val res = LogisticRegression.classifyVector(DenseVector(row._1), curr._1)
      val err = if (labels(row._2) != res) curr._2 + 1
      else curr._2
      (curr._1, err)
    })._2
    println(error2.toDouble / dataSet.length)
    println()
    val error3 = dataSet.zipWithIndex.foldLeft((LogisticRegression.stocGradAscent1(dataSet, labels), 0))((curr, row) => {
      val res = LogisticRegression.classifyVector(DenseVector(row._1), curr._1)
      val err = if (labels(row._2) != res) curr._2 + 1
      else curr._2
      (curr._1, err)
    })._2
    println(error3.toDouble / dataSet.length)    
  }
}