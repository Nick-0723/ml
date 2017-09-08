package mlia_breeze.classifier.bayes
import scala.io.Source

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

object EmailTest extends App {
  def textParser(bigString: String) = {
    "\\W".r.split(bigString).filterNot(x => x.trim.length() >= 2).map(_.toLowerCase())
  }
  def createVocabList(dataSet: DenseVector[String]): Array[String] =
    (0 until dataSet.length).foldLeft(Array.empty[String])((set, i) => dataSet(i).split(" ") ++ set).distinct

  def setOfWordsToVec(vocabList: Array[String], inputStr: String): DenseVector[Int] = {
    val numWords = vocabList.size
    val returnVec = DenseVector.zeros[Int](numWords)
    val f = textParser(inputStr)
    f.foreach(x => returnVec(vocabList.indexOf(x)) = 1)
    returnVec
  }
  
  override def main(args: Array[String]): Unit = {
    val (docList, fullList, classList) = (1 until 26).foldLeft((Array.empty[String], Array.empty[String], Array.empty[Int]))((list, index) => {
      val input = Source.fromFile(s"email/spam/${index}.txt").getLines().toArray
      val wordsList = input.flatMap(x => textParser(x))
      var docList = list._1 ++ Array(input.mkString(" "))
      var fullList = list._2 ++ wordsList
      var classList = list._3 ++ Array(1)
      val input1 = Source.fromFile(s"email/ham/${index}.txt").getLines().toArray
      val wordsList1 = input1.flatMap(x => textParser(x))
      docList = docList ++ Array(input1.mkString(" "))
      fullList = fullList ++ wordsList1
      classList = classList ++ Array(0)
      (docList, fullList, classList)
    })
    val vocabList = createVocabList(DenseVector(docList))
    vocabList.foreach(x => print(x + ","))
    println

    val testSet = (0 until 10).foldLeft((Array.emptyIntArray, new util.Random))((vec, random) => {
      (vec._1 :+ vec._2.nextInt(50), vec._2)
    })._1.distinct

    val trainingSet = (Array.emptyIntArray ++ (0 until 50)).filterNot(x => testSet.contains(x))

    var index = 0
    val s1 = trainingSet.map(docList(_))
    val s2 = s1.map(x => setOfWordsToVec(vocabList, x))

    val trainMat = s2.foldLeft(DenseMatrix.zeros[Int](s2.length, vocabList.length))((mat, row) => {
      mat(index, ::) := row.t
      index = index + 1
      mat
    })
    val trainClasses = DenseVector(trainingSet.map(x => classList(x))) 
    println("testSet:" + testSet.size)

    println("trainingSet:" + trainingSet.size)

    println("trainingClasses:" + trainClasses.size)
    println("trainingMatrix:" + trainMat.rows)
   
    val trainMatrix = NativeBayes.trainNB0(trainMat, trainClasses)
    
    val error = testSet.foldLeft(0)((error, str) => {
      val strTokens = setOfWordsToVec(vocabList, docList(str))
      val res = NativeBayes.classifyNB(strTokens, trainMatrix._1.logProbability, trainMatrix._2.logProbability, trainMatrix._3)
      if(res != classList(str)) error + 1
      else error
    })
    println(error)
    println(testSet.length)
    println("the error rate is: " + error/testSet.length.toDouble)
  }
}