package mlia_breeze.filter

import scala.None

object Apriori {
  case class ItemSet(x: Set[Int]) extends Set[Int]{
    override def toString(): String = s"Item[${x.mkString(", ")}]"
    override def iterator: Iterator[Int] = x.iterator
    override def +(e: Int) = x + e
    override def -(e: Int) = x - e
    override def contains(e: Int) = x.contains(e)
    def toH1 = x.map(ItemSet(_)).toArray
  }

  object ItemSet{
    def apply(e: Int) = new ItemSet(Set(e))
    def apply(arr: Array[Int]) = new ItemSet(arr.toSet)

    implicit def set2ItemSet(ds: Set[Int]): ItemSet = ItemSet(ds)
  }

  case class Supports(map: Map[ItemSet, Double] = Map.empty[ItemSet, Double]) {
    override def toString: String = s"Supports:\n${map.mkString("\n")}"
    def +(kv: (ItemSet, Double)) = new Supports(map + kv)
    def ++(supp: Supports) = new Supports(map ++ supp.map)

    def apply(set: ItemSet): Double = map(set)
  }

  object Supports{
    def empty = new Supports()
  }

  def createC1(dataSet: Array[Array[Int]]): Array[ItemSet] = dataSet.flatten.distinct.sorted.map(ItemSet(_))

  def scanD(D: Array[ItemSet], Ck: Array[ItemSet], minSupport: Double):(Array[ItemSet], Supports) = {
    D.foldLeft(Map.empty[ItemSet, Int]){
      (outer, tran) =>
        Ck.foldLeft(outer){
          (inner, item) =>
            if(item.subsetOf(tran))inner + (item -> (inner.getOrElse(item, 0) + 1)) else inner
        }
    }.foldLeft((Array.empty[ItemSet], Supports.empty)){
      case ((retValues, data), (can, cnt)) =>
        val v = cnt.toDouble / D.length
        val is: Array[ItemSet] = if(v >= minSupport) can +: retValues else retValues
        (is, (data + (can -> v)))
    }
  }

  def aprioriGen(Lk: Array[ItemSet], k:Int): Array[ItemSet] = {
    val res: Array[ItemSet] = (for{(x1, i) <- Lk.zipWithIndex
                                   (x2, j) <- Lk.zipWithIndex
                                   if i > j
                                   if(x1.toArray.take(k - 2).toSet == x2.toArray.take(k - 2).toSet)

    } yield x1.union(x2)).map( ItemSet.apply )
    res
/*   Lk.flatMap(x1 => Lk.map(x2 => {
     if (x1.toArray.sorted.take(k - 2).toSet == x2.toArray.sorted.take(k - 2).toSet)
     x1.union(x2).map(ItemSet(_))
     else Set.empty[ItemSet]
   })).filter(x => x.size == k)
   */
 }

 def apriori(dataSet: Array[Array[Int]], minSupport: Double = 0.5): (Array[Array[ItemSet]], Supports) = {
   val C1 = createC1(dataSet)
   val D = dataSet.map(ItemSet.apply)
   val (l1, support) = scanD(D, C1, minSupport)

   def loop(curL: Array[Array[ItemSet]], curSupport: Supports, curK: Int = 2):(Array[Array[ItemSet]], Supports) = {
     if(curL(curK - 2).isEmpty)(curL, curSupport)
     else{
       val Ck = aprioriGen(curL(curK - 2), curK)
       val (lk, supp) = scanD(D, Ck, minSupport)
       loop(curL :+ lk, curSupport ++ supp, curK + 1)
     }
   }
   loop(Array(l1), support)
 }

  case class Rule(leftSide: ItemSet, rightSide: ItemSet, confidence: Double){
    override def toString() = f"[${leftSide.mkString(",")}] ---> [${rightSide.mkString(",")}] : confidence:$confidence"

  }

  implicit def array2ItemSet(ds: Array[Array[Int]]): Array[ItemSet] = ds.map(x => ItemSet(x.toSet))

  def calcConf(freqSet: ItemSet, H: Array[ItemSet], supportData: Supports, minConf: Double):Array[Rule] = {
    H.map(conseq => (conseq, supportData(freqSet) / supportData(freqSet -- conseq)))
      .filter(_._2 > minConf)
      .map{
      case (conseq, conf) =>
        Rule(freqSet -- conseq, conseq, conf)
      }
  }

  def rulesFromConseq(freqSet: ItemSet, H: Array[ItemSet],
                      supportData: Supports, minConf: Double,
                      state: Array[Rule] = Array.empty):Array[Rule] = {
    val m = H(0).size
    if(freqSet.size <= m + 1) state
    else {
      val Hmp1 = state ++ calcConf(freqSet, aprioriGen(H, m+1), supportData, minConf)
      val prunedH = Hmp1.map(_.rightSide)
      if(prunedH.size > 1){
        Hmp1 ++ rulesFromConseq(freqSet, prunedH, supportData, minConf, Hmp1)
      }else Hmp1
    }
  }

  def generateRules(ls: Array[Array[ItemSet]], supportData: Supports, minConf: Double = 0.7) ={
    ls.drop(1).flatMap{
      case items =>
        items.flatMap{
          freqSet =>
            if(freqSet.size <= 2)calcConf(freqSet, freqSet.toH1, supportData, minConf)
            else rulesFromConseq(freqSet, freqSet.toH1, supportData, minConf)
        }
    }
  }
}
