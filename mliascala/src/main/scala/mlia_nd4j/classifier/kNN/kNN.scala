package mlia_nd4j.classifier.kNN

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

object kNN {
  def classify0(inX: INDArray, dataSet: INDArray, labels: Array[String], k: Int): String = {
    require(k > 0)
    require(dataSet.rows == labels.length)
    require(inX.length() == dataSet.columns())
    val distances = (0 until dataSet.rows()).foldLeft(Nd4j.create(dataSet.rows(), 1)){
      (res, idx) =>{
        res(idx, 0) = dataSet(idx, ->).distance2(inX)
        res
      }
    }
    val sortedDistIndicies: Array[INDArray] = Nd4j.sortWithIndices(distances, 0, true)
    val indecs: Array[Int] = sortedDistIndicies(0).data().asInt().take(k)

    val classCount  = indecs.foldLeft(Map.empty[String, Int]){
      (counter, idx) =>
        val vote = labels(idx)
        counter + (vote -> (counter.getOrElse(vote, 0) + 1))
    }
    classCount.toArray.sortBy(_._2).reverse.headOption.map(_._1).getOrElse("Failure")
  }
  def autoNorm(dataSet: INDArray): (INDArray, INDArray, INDArray) = {

    val minVals = (0 until dataSet.columns()).foldLeft(Nd4j.create(1, dataSet.columns())){
      (array, idx) =>
        Transforms.min(dataSet(->, idx), 1)
    }
    val maxVals = (0 until dataSet.columns()).foldLeft(Nd4j.create(1, dataSet.columns())){
      (array, idx) =>
        Transforms.max(dataSet(->, idx), 1)
    }

    val ranges = maxVals - minVals
    val diff = dataSet - minVals.reshape(minVals.rows(), minVals.columns())//tile1(minVals, m.rows)
    (diff / diff, ranges, minVals)

  }
  def createDataSet  = {
    val dataSet = Nd4j.create(Array[Float](1.0f, 1.1f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.1f), Array[Int](4, 2))
    val labels = Array[String]("A", "A", "B", "B")
     (dataSet, labels)
  }

  def main(args: Array[String]): Unit = {
    val (dataSet, labels)  = createDataSet
    val inX = Nd4j.create(Array[Float](0.0f, 1.1f), Array[Int](1, 2))
    printf("classify: %s\n", classify0(inX, dataSet, labels, 1))
  }
}
