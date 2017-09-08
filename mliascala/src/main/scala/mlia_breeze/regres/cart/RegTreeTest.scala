package mlia_breeze.regres.cart

import breeze.linalg.DenseVector

object RegTreeTest extends App {
  override def main(args: Array[String]):Unit = {
        val  inData  = Prep.loadDataSet("cart/ex2.txt")
        val dataSet = DataSet(inData)
        val reg = dataSet.createTree(Array(0, 1))(Cart.Regression)
        val testData = Prep.loadDataSet("cart/ex2test.txt")
     //   val model = dataSet.createTree(Array(1.0, 4.0))(Cart.Model)
        println(reg)
        val r = reg.prune(DataSet(testData)) 
        println(r)
        
  }
}