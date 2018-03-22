import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, SparkSession}

object Classification {

  case class Message(spam: String, message: String)

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("classification")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val path =
      "/Users/mreiss/dev/scala/sparkml/src/main/resources/SMSSpamCollection.csv"
    val data: Dataset[Message] =
      spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .as[Message]

    //    data.createOrReplaceTempView("data")
    //    val result = spark.sql("SELECT * FROM data")

    data.printSchema()
    data.show(7)

    val spamIndexer =
      new StringIndexer().setInputCol("spam").setOutputCol("label")

    val tokenizer =
      new Tokenizer().setInputCol("message").setOutputCol("messageToken")

    val hashingTF = new HashingTF()
      .setInputCol("messageToken")
      .setOutputCol("messageHF")

    val assembler = new VectorAssembler()
      .setInputCols(Array("messageHF"))
      .setOutputCol("features")

    // Split the Data
    val Array(training, test) = data.randomSplit(Array(0.75, 0.25), seed = 43)

    // Set Up the Pipeline
    val lr = new LogisticRegression()

    val pipeline = new Pipeline()
      .setStages(Array(spamIndexer, tokenizer, hashingTF, assembler, lr))

    // build the model
    val model: PipelineModel = pipeline.fit(training)

    // predict
    val results = model.transform(test)
    results.show(10)

    // evaluate
    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(results)
    println(s"Accuracy = ${accuracy}")

    // compute confusion matrix
    val predictionsAndLabels = results
      .select("prediction", "label")
      .map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionsAndLabels.rdd)
    println("\nConfusion matrix:")
    println(metrics.confusionMatrix)

    spark.stop()
  }
}
