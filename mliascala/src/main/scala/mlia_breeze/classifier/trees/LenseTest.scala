package mlia_breeze.classifier.trees

import scala.io.Source

object LenseTest extends App{
  def parser(line: String): Tree.Row = {
    val tokens = line.trim.split(" ").filter(!_.trim.equals(""))
    Tree.Row(Array(tokens(1).toInt, tokens(2).toInt, tokens(3).toInt, tokens(4).toInt), tokens(5))
  }
  
  def fileToDataSet(fileName: String): Array[Tree.Row] = {
          Source.fromFile(fileName).getLines().toArray.map(parser(_))  
     
  }
  override def main(args: Array[String]): Unit = {
    val dataSet = fileToDataSet("lenses/lenses.data.txt")
    dataSet.foreach(println(_))
    val featureLabels = Array("age", "prescript", "astigmatic", "tearRate")
    val tree = Tree(dataSet, featureLabels)
    println(tree)
  }
}