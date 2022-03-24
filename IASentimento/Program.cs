using IASentimento;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace ExemploIASentimento
{
    class Program
    {
        static void Main(string[] args)
        {
            string _dataPath = Path.Combine("C: \\Users\\Gabriel\\Desktop\\IASentimento", "ClassificacaoVinhos2.csv");
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext, _dataPath);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            IEnumerable<VinhoData> winetests = new[]
            {
                 new VinhoData() {
                fixedacidity = 6.6,
                    volatileacidity = 0.17,
                    citricacid = 0.38,
                    residualsugar = 1.5,
                    chlorides = 0.032,
                    freesulfurdioxide = 28,
                    totalsulfurdioxide = 112,
                    density=0.9914,
                    pH = 3.25,
                    sulphates=0.55,
                    alcohol=11.4,
                    quality= 7,
                    Classificacao= false
                 },
                 new VinhoData() {
                    fixedacidity = 7,
                    volatileacidity = 0.31,
                    citricacid = 0.26,
                    residualsugar = 7.4,
                    chlorides = 0.069,
                    freesulfurdioxide = 28,
                    totalsulfurdioxide = 160,
                    density=0.9954,
                    pH = 3.13,
                    sulphates=0.46,
                    alcohol=9.8,
                    quality= 6,
                    Classificacao= false
                  },
                 new VinhoData() {
                    fixedacidity = 6.3,
                    volatileacidity = 0.3,
                    citricacid = 0.34,
                    residualsugar = 1.6,
                    chlorides = 0.049,
                    freesulfurdioxide = 14,
                    totalsulfurdioxide = 132,
                    density=0.994,
                    pH = 3.3,
                    sulphates=0.49,
                    alcohol=9.5,
                    quality= 6,
                    Classificacao= false
                  },

                 new VinhoData() {
                    fixedacidity = 9.6,
                    volatileacidity = 0.68,
                    citricacid = 0.24,
                    residualsugar = 2.2,
                    chlorides = 0.087,
                    freesulfurdioxide = 5,
                    totalsulfurdioxide = 28,
                    density=0.9988,
                    pH = 3.14,
                    sulphates=0.6,
                    alcohol=1025,
                    quality= 5,
                    Classificacao= true
                  },

                 new VinhoData() {
                    fixedacidity = 9.3,
                    volatileacidity = 0.27,
                    citricacid = 0.41,
                    residualsugar = 2,
                    chlorides = 0.091,
                    freesulfurdioxide = 6,
                    totalsulfurdioxide = 16,
                    density=0.998,
                    pH = 3.28,
                    sulphates=0.7,
                    alcohol=9.7,
                    quality= 5,
                    Classificacao= true
                  }

           };
            //7;0.27;0.36;20.7;0.045;45;170;1.001;3;0.45;8.8;6
            UseModelWithBatchItems(mlContext, model, winetests);

        }
        private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            throw new NotImplementedException();
        }


        public static TrainTestData LoadData(MLContext mlContext, string _dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<VinhoData>(_dataPath, hasHeader: true);

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;

        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet, ITransformer model)
        {

            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Features", inputColumnName: nameof(VinhoData.fixedacidity))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_volatile", outputColumnName: "VinhoData.volatileacidity"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_citric", outputColumnName: "VinhoData.citricacid"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_residual", outputColumnName: "VinhoData.residualsugar"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_chlorides", outputColumnName: "VinhoData.chlorides"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_free", outputColumnName: "VinhoData.freesulfurdioxide"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_total", outputColumnName: "VinhoData.totalsulfurdioxide"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_density", outputColumnName: "VinhoData.density"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_pH", outputColumnName: "VinhoData.pH"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_sulphates", outputColumnName: "VinhoData.sulphates"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "veil_alcohol", outputColumnName: "VinhoData.alcohol"))
                             .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "veil_quality", outputColumnName: "VinhoData.quality"))
                             .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[] { "VinhoData.volatileacidity", "VinhoData.citricacid"
                                                                                   , "VinhoData.residualsugar", "VinhoData.chlorides", "VinhoData.freesulfurdioxide", "VinhoData.totalsulfurdioxide"
                                                                                   , "VinhoData.density", "VinhoData.pH", "VinhoData.sulphates", "VinhoData.alcohol"
                                                                                   , "VinhoData.quality" }));
            return model;
        }

        private static void UseModelWithBatchItems(MLContext mlContext, ITransformer model, IEnumerable<VinhoData> wineTests)
        {
            IDataView listaVinhosTeste = mlContext.Data.LoadFromEnumerable(wineTests);

            IDataView predictions = model.Transform(listaVinhosTeste);

            IEnumerable<VinhoPrediction> predictedResults =
                mlContext.Data.CreateEnumerable<VinhoPrediction>(predictions, reuseRowObject: false);

            int nroTeste = 1;
            foreach (VinhoPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Teste {nroTeste} : {(Convert.ToBoolean(prediction.Prediction) ? "Vermelho" : "Branco")} | Probability: {prediction.Probability} ");
                nroTeste++;
            }
            Console.WriteLine("=============== FIM ===============");
        }

    }
}