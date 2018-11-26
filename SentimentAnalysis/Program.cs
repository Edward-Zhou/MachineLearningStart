using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("SentimentAnalysis Start!");
            //1. Create ML.NET context/environment
            MLContext mLContext = new MLContext();
            //2. Create DataReader with data schema mapped to file's columns
            string baseDataPath = @"Data/base.tsv";

            var reader = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new TextLoader.Column[] {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("Text", DataKind.Text, 1)
                }
            };
            //Load training data
            IDataView trainingDataView = mLContext.Data.TextReader(reader).Read(new MultiFileSource(baseDataPath));

            //3.Create a flexible pipeline (composed by a chain of estimators) for creating/traing the model.
            var pipeline = mLContext.Transforms.Text.FeaturizeText("Text", "Features")
                                    .Append(mLContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeafs: 20));
            //Train model
            var model = pipeline.Fit(trainingDataView);

            //Evaluate model
            var testDataPath = @"Data/test.tsv";
            IDataView testDataView = mLContext.Data.TextReader(reader).Read(new MultiFileSource(testDataPath));
            var predictions = model.Transform(testDataView);
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            Console.ReadLine();

            //Save Model
            using (var stream = new FileStream(@"Data/model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mLContext.Model.Save(model, stream);
            }
            //Consume model
            var predictionFunct = model.MakePredictionFunction<SentimentIssue, SentimentPrediction>(mLContext);
            var sampleStatement = new SentimentIssue
            {
                Text = "This is a very rude movie"
            };
            var resultprediction = predictionFunct.Predict(sampleStatement);

            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(resultprediction.Prediction ? "Negative" : "Positive")} sentiment");
            Console.ReadLine();
        }
    }
}
