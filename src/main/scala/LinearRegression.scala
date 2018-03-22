import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{
  OneHotEncoder,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{abs, col}

object LinearRegression {
  case class HousingData(id: String,
                         price: Double,
                         lotsize: Double,
                         bedrooms: Double,
                         bathrooms: Double,
                         stories: Double,
                         driveway: String,
                         recRoom: String,
                         finishedBasement: String,
                         gas: String,
                         ac: String,
                         garages: Double,
                         goodArea: String)

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val path = getClass.getResource("Housing.csv").getPath

    val spark = SparkSession.builder
      .appName("linearregression")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val housingData = spark.sparkContext
      .textFile(path)
      .map(_.split(","))
      .map(
        x =>
          HousingData(x(0),
                      x(1).toDouble,
                      x(2).toDouble,
                      x(3).toDouble,
                      x(4).toDouble,
                      x(5).toDouble,
                      x(6),
                      x(7),
                      x(8),
                      x(9),
                      x(10),
                      x(11).toDouble,
                      x(12)))
    val data = housingData.toDF().as[HousingData]

    data.show(10)

    // Define and Identify the Categorical variables
    val categoricalVariables =
      Array("driveway", "recRoom", "finishedBasement", "gas", "ac", "goodArea")

    //Initialize the Categorical Varaibles as first state of the pipeline
    val categoricalIndexers: Array[PipelineStage] =
      categoricalVariables.map(
        i =>
          new StringIndexer()
            .setInputCol(i)
            .setOutputCol(i + "Index"))

    // Initialize the OneHotEncoder as another pipeline stage
    val categoricalEncoders: Array[PipelineStage] =
      categoricalVariables.map(
        e =>
          new OneHotEncoder()
            .setInputCol(e + "Index")
            .setOutputCol(e + "Vec"))

    // Put all the feature columns of the categorical variables together
    val assembler = new VectorAssembler()
      .setInputCols(
        Array("lotsize",
              "bedrooms",
              "bathrooms",
              "stories",
              "garages",
              "drivewayVec",
              "recRoomVec",
              "finishedBasementVec",
              "gasVec",
              "acVec",
              "goodAreaVec"))
      .setOutputCol("features")

    // Initialize the instance for LinearRegression using your choice of solver and number of iterations
    // Experiment with intercepts and different values of regularization parameter
    val lr = new LinearRegression()
      .setLabelCol("price")
      .setFeaturesCol("features")
      .setRegParam(0.1)

    // Gather the steps and create the pipeline
    val steps = categoricalIndexers ++
      categoricalEncoders ++
      Array(assembler, lr)

    val pipeline = new Pipeline().setStages(steps)

    // Split the data into training and test
    val Array(training, test) =
      data.randomSplit(Array(0.75, 0.25), seed = 42)

    // Fit the model and print out the result
    val model: PipelineModel = pipeline.fit(training)

    val results = model.transform(test)
    results.show(10)

    val prediction = results
      .select("prediction", "price")
      .orderBy(abs(col("prediction") - col("price")))
    prediction.show(10)
    prediction.orderBy(abs(col("prediction") - col("price")).desc).show(20)

    val rm = new RegressionMetrics(prediction.rdd.map { x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])
    })
    println(s"RMSE = ${rm.rootMeanSquaredError}")
    println(s"R-squared = ${rm.r2}")

    spark.stop()
  }

}
