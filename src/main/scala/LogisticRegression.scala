import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val path = getClass.getResource("titanic.csv").getPath

    val spark = SparkSession
      .builder()
      .appName("logisticregression")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(path)

    data.printSchema()
    data.show(8)

    // Grab only the columns we want
    val logRegDataAll = data.select(data("Survived").as("label"),
                                    $"Pclass",
                                    $"Sex",
                                    $"Age",
                                    $"SibSp",
                                    $"Parch",
                                    $"Fare",
                                    $"Embarked")
    val logRegData = logRegDataAll.na.drop()
    logRegData.show(8)

    // Deal with Categorical Columns
    val genderIndexer: PipelineStage =
      new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer: PipelineStage =
      new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

    val genderEncoder: PipelineStage =
      new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
    val embarkEncoder: PipelineStage =
      new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

    // Assemble everything together to be ("label","features") format
    val assembler = new VectorAssembler()
      .setInputCols(
        Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec"))
      .setOutputCol("features")

    // Split the Data
    val Array(training, test) =
      logRegData.randomSplit(Array(0.75, 0.25), seed = 42)

    // Set Up the Pipeline
    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(
      Array(genderIndexer,
            embarkIndexer,
            genderEncoder,
            embarkEncoder,
            assembler,
            lr))

    // Fit the pipeline to training documents.
    val model: PipelineModel = pipeline.fit(training)

    // Get Results on Test Set
    val results = model.transform(test)
    results.show(10)

    // evaluate
    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(results)
    println(s"Accuracy = $accuracy")

    // Need to convert to RDD to use MulticlassMetrics
    val predictionAndLabels =
      results.select($"prediction", $"label").as[(Double, Double)].rdd

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
  }
}
