package mlia_breeze.classifier.trees

import breeze.linalg.DenseVector

object EntroTest extends App {

  def createDataSet(): Array[Tree.Row] = {
    val dataSet = Array.empty[Tree.Row]
    val s1 = Tree.Row(Array(1, 1), "yes") +: dataSet
    val s2 = Tree.Row(Array(1, 1), "yes") +: s1
    val s3 = Tree.Row(Array(1, 0), "no") +: s2
    val s4 = Tree.Row(Array(0, 1), "no") +: s3
    val s5 = Tree.Row(Array(0, 1), "no") +: s4
    s5
  }
  override def main(args: Array[String]): Unit = {
    val dataSet = createDataSet()
    println(Tree.calcShannonEnt(dataSet))
    Tree.splitDataSet(dataSet, 0, 1).foreach(x => printf("[%d, %s]", x.data(1), x.label))
    println()
    Tree.splitDataSet(dataSet, 0, 0).foreach(x => printf("[%d, %s]", x.data(1), x.label))
    println()

    println(Tree.chooseBestFeatureToSplitData(dataSet))
    val labels = Array("no surfacing","flippers")//dataSet.map(_.label)
    val descTree = Tree(dataSet, labels)
    println(descTree)
    
    println(descTree.classfy(DenseVector(1, 0), Array("no surfacing","flippers")))
        println(descTree.classfy(DenseVector(1, 1), Array("no surfacing","flippers")))

  }
}