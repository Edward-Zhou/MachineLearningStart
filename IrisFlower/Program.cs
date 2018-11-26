using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;

namespace IrisFlower
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ML Start!");

            MLContext context = new MLContext();
            string dataPath = "iris-data.txt";
            var reader = context.Data.TextReader(new TextLoader.Arguments() {
                Separator = ",",
                HasHeader = true,
                Column  = new[] {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            });
            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));

            var pipeline = context.Transforms.Categorical.MapValueToKey("Label")
                .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipeline.Fit(trainingDataView);

            var prediction = model.MakePredictionFunction<IrisData, IrisPrediction>(context)
                .Predict(new IrisData {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f
                });
            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
    }
}
