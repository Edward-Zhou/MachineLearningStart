using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    public class SentimentIssue
    {
        [Column("Label")]
        public float Label { get; set; }
        [Column("Text")]
        public string Text { get; set; }
    }
}
