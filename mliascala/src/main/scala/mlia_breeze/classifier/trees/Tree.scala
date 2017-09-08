package mlia_breeze.classifier.trees
import breeze.linalg._
import breeze.stats.distributions._
import breeze.numerics._
import scala.annotation.tailrec

case class Node(key: String, value: Any, children: Array[Node] = Array.empty){
  val isLeaf = children.isEmpty
  
  override def toString = 
    if (children.isEmpty) s" -> $value[Leaf]" 
    else s"{$key : $value ${children.map(_.toString).mkString(",")}}"
}

case class Tree(nodes: Array[Node] = Array.empty){
  override def toString = s"Tree[${nodes.map(_.toString).mkString(",")}]"
  
  def <<-(node: Node): Tree = Tree(nodes :+ node)
  
  def classfy(testVec: Vector[Int], featLabels: Array[String], cur: Array[Node] = nodes): String = search(testVec, featLabels, nodes)

  
  @tailrec
  private def search(testVec: Vector[Int], featLabels: Array[String], cur: Array[Node]): String = {
    cur.find(node =>{
      node.isLeaf || testVec(featLabels.indexOf(node.key)).toString == node.value.toString
    }) match {
      case None => "Fail to classify"
      case Some(node) if node.isLeaf => node.value.toString
      case Some(node) => search(testVec, featLabels, node.children)
    }
  }
}
object Tree {
  case class Row(data: Array[Int], label: String)
  
  def calcShannonEnt(dataSet: Array[Row]): Double = {
     val numEntries = dataSet.size
     val labelCounts = dataSet.foldLeft(Map.empty[String, Int])((map, row) => {
         map + (row.label -> (map.getOrElse(row.label, 0) + 1))
     })
    
    labelCounts.foldLeft(0.0)((state, count) =>{
      val prob = labelCounts(count._1).toDouble / numEntries
      state - prob * (log(prob) / log(2))
    })
  }
  
  def splitDataSet(dataSet: Array[Row], axis: Int, value: Int) = {
    dataSet.filter(x => x.data(axis) == value)
  }
  case class InformationGain(featureIdx: Int, infoGain: Double)

  def chooseBestFeatureToSplitData(dataSet: Array[Row]): InformationGain = {
    val baseEntropy = calcShannonEnt(dataSet)
    val numEntries = dataSet.size
    val numFeatures  = dataSet.head.data.size
    (0 until numFeatures).foldLeft(InformationGain(-1, 0.0))((curBest, cur) => {
      val uniqueVals = dataSet.map(x => x.data(cur)).distinct
      val newEntropy = uniqueVals.foldLeft(0.0)((ent,  value) => {
        val subDataSet = splitDataSet(dataSet, cur, value)
        val prob = subDataSet.size / numEntries.toDouble
        ent + prob * calcShannonEnt(subDataSet)
      })
      val infoGain = baseEntropy - newEntropy
      if (infoGain > curBest.infoGain) InformationGain(cur, infoGain) else curBest
    })
  }
  
  def majorityCnt(classList: Array[String]): String = {
    classList.foldLeft(Map.empty[String, Int])((map, vote) => {
      map + (vote -> (map.getOrElse(vote, 0) + 1))
    }).toArray.sortBy(_._2).reverse.headOption.map(_._1).getOrElse("failure")
  }
  
  private def remove(num:Int, list:Array[String]) = list.diff(Array(num))
  
  def apply(dataSet: Array[Row], labels: Array[String]) = createTree(dataSet, labels)
  
  def createTree(dataSet: Array[Row], labels: Array[String], cur: Tree = Tree(), value: Int = -1) : Tree= {
    val classList = dataSet.map(_.label)
    
    if(classList.distinct.size == 1) cur <<- Node(value.toString, classList(0))
    else if(dataSet.head.data.isEmpty) cur <<- Node(value.toString, majorityCnt(classList))
    else {
      val bestFeat = chooseBestFeatureToSplitData(dataSet).featureIdx
      val subLabels = remove(bestFeat, labels)
      val uniqueFeatValues = dataSet.map(_.data(bestFeat)).distinct
      uniqueFeatValues.foldLeft(cur)((state, value) => {
        val subTree = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, cur, value)
        state <<- Node(labels(bestFeat), value.toString, subTree.nodes)
      })
    }
  }
}