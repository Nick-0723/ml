package mlia_breeze.classifier.kNN
import breeze.linalg._
import breeze.stats.distributions._
import breeze.numerics._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
object kNN {
  def classify0(inX: Vector[Double], dataSet: Matrix[Double], labels: Vector[String], k: Int): String = {
    require(k > 0)
    require(dataSet.rows == labels.size)
    require(inX.size == dataSet.cols)

    val diffMat = tile1(inX.toDenseVector, dataSet.rows) - dataSet.toDenseMatrix
    val sqDiffMat = diffMat :^ 2.0 //.map(x => x * x)
    val sqDistances = sum(sqDiffMat(*, ::))
    val distances = sqrt(sqDistances) //sqDistances.map(x => math.sqrt(x))
    val sortedDistIndicies = distances.activeIterator.toArray.sortBy(_._2).take(k)
    val classCount = sortedDistIndicies.foldLeft(Map.empty[String, Int]) { (map, dist) =>
      val vote = labels(dist._1)
      map + (vote -> (map.getOrElse(vote, 0) + 1))

    }
    classCount.toArray.sortBy(_._2).reverse.headOption.map(_._1).getOrElse("Failure")

  }

  private def tile1(in: Vector[Double], repeat: Int): DenseMatrix[Double] = {
    val mat = DenseMatrix.zeros[Double](repeat, in.size)
    for (i <- 1 to repeat) mat(i - 1, ::) := in.t
    mat
  }

  def autoNorm(dataSet: Matrix[Double]): (Matrix[Double], Vector[Double], Vector[Double]) = {
    val m = dataSet.toDenseMatrix

    val minVals = min(m(::, *)).t
    val maxVals = max(m(::, *)).t

    val ranges = maxVals - minVals
    val diff = m - tile1(minVals, m.rows)
    (diff /:/ tile1(ranges, diff.rows), ranges, minVals)

  }

}