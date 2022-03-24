using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace IASentimento
{


        // <SnippetDeclareTypes>
        public class VinhoData
        {
            [LoadColumn(0)]
            public double fixedacidity;

            [LoadColumn(1)]
            public double volatileacidity;

            [LoadColumn(2)]
            public double citricacid;

            [LoadColumn(3)]
            public double residualsugar;

            [LoadColumn(4)]
            public double chlorides;

            [LoadColumn(5)]
            public double freesulfurdioxide;

            [LoadColumn(6)]
            public double totalsulfurdioxide;

            [LoadColumn(7)]
            public double density;

            [LoadColumn(8)]
            public double pH;

            [LoadColumn(9)]
            public double sulphates;

            [LoadColumn(10)]
            public double alcohol;

            [LoadColumn(11)]
            public double quality;

            [LoadColumn(12), ColumnName("Label")]
            public bool Classificacao;
        }
    public class VinhoPrediction : VinhoData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }

}