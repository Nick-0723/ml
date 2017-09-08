package mlia_breeze.filter

import mlia_breeze.filter.Apriori.ItemSet

object AprioriTest extends  App{
  def loadDate() = Array(Array(1,3,4), Array(2,3,5), Array(1,2,3,5), Array(2,5))
  override def main(args: Array[String]) = {
    val dataSet = loadDate()
    val c1 = Apriori.createC1(dataSet)
    c1.foreach(println)
    val D = dataSet.map(x => new ItemSet(x.toSet))
    D.foreach(println)
    val (l1, supp1) = Apriori.scanD(D, c1, 0.5)
    l1.foreach(println)

    val sss: Array[ ItemSet ] = Apriori.aprioriGen(l1, 2)
    sss.foreach(println)

    val (lk, supp)  = Apriori.apriori(dataSet)
    lk.foreach(x => x.foreach(println))
    println()
    val rules = Apriori.generateRules(lk, supp, minConf = 0.6)
    rules.foreach(println)
   }
}
