import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, AlertTriangle, BarChart3 } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ModelPerformanceCard = ({ title, metrics, chartData }) => {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="p-2 bg-blue-50 rounded">
              <p className="text-sm text-gray-600">MAE</p>
              <p className="text-lg font-bold">{metrics.mae.toFixed(2)}</p>
            </div>
            <div className="p-2 bg-blue-50 rounded">
              <p className="text-sm text-gray-600">RMSE</p>
              <p className="text-lg font-bold">{metrics.rmse.toFixed(2)}</p>
            </div>
            <div className="p-2 bg-blue-50 rounded">
              <p className="text-sm text-gray-600">R²</p>
              <p className="text-lg font-bold">{metrics.r2.toFixed(2)}</p>
            </div>
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

const DataSplitVisualization = ({ trainSize, testSize, validationSize }) => {
  const total = trainSize + testSize + validationSize;
  const trainPercent = (trainSize / total) * 100;
  const testPercent = (testSize / total) * 100;
  const validationPercent = (validationSize / total) * 100;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">Data Split Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-8 flex rounded-lg overflow-hidden">
          <div 
            className="bg-blue-500 h-full" 
            style={{ width: `${trainPercent}%` }}
          >
            <div className="text-white text-xs p-1">Train ({trainPercent.toFixed(1)}%)</div>
          </div>
          <div 
            className="bg-green-500 h-full" 
            style={{ width: `${testPercent}%` }}
          >
            <div className="text-white text-xs p-1">Test ({testPercent.toFixed(1)}%)</div>
          </div>
          <div 
            className="bg-yellow-500 h-full" 
            style={{ width: `${validationPercent}%` }}
          >
            <div className="text-white text-xs p-1">Val ({validationPercent.toFixed(1)}%)</div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-600">Training Samples</p>
            <p className="text-lg font-bold">{trainSize.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Test Samples</p>
            <p className="text-lg font-bold">{testSize.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600">Validation Samples</p>
            <p className="text-lg font-bold">{validationSize.toLocaleString()}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const FileUpload = ({ onUpload }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/upload/training-data', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      onUpload(data);
    } catch (err) {
      setError('Failed to upload file. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">Upload Training Data</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center w-full">
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-8 h-8 mb-4 text-gray-500" />
              <p className="mb-2 text-sm text-gray-500">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">CSV files only</p>
            </div>
            <input 
              type="file" 
              className="hidden" 
              accept=".csv"
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </label>
        </div>
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

const SupplyChainDashboard = () => {
  const [modelData, setModelData] = useState(null);
  const [dataSplit, setDataSplit] = useState(null);

  const handleDataUpload = (data) => {
    // Example response data structure
    setModelData({
      demandForecaster: {
        metrics: {
          mae: 12.34,
          rmse: 15.67,
          r2: 0.89
        },
        chartData: data.demandPredictions
      },
      inventoryOptimizer: {
        metrics: {
          mae: 8.45,
          rmse: 10.23,
          r2: 0.92
        },
        chartData: data.inventoryPredictions
      },
      priceOptimizer: {
        metrics: {
          mae: 5.67,
          rmse: 7.89,
          r2: 0.95
        },
        chartData: data.pricePredictions
      }
    });

    setDataSplit({
      trainSize: data.splits.train,
      testSize: data.splits.test,
      validationSize: data.splits.validation
    });
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Supply Chain Optimization Dashboard</h1>
      
      <div className="mb-8">
        <FileUpload onUpload={handleDataUpload} />
      </div>

      {dataSplit && (
        <div className="mb-8">
          <DataSplitVisualization {...dataSplit} />
        </div>
      )}

      {modelData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <ModelPerformanceCard 
            title="Demand Forecaster"
            metrics={modelData.demandForecaster.metrics}
            chartData={modelData.demandForecaster.chartData}
          />
          <ModelPerformanceCard 
            title="Inventory Optimizer"
            metrics={modelData.inventoryOptimizer.metrics}
            chartData={modelData.inventoryOptimizer.chartData}
          />
          <ModelPerformanceCard 
            title="Price Optimizer"
            metrics={modelData.priceOptimizer.metrics}
            chartData={modelData.priceOptimizer.chartData}
          />
        </div>
      )}
    </div>
  );
};

export default SupplyChainDashboard;