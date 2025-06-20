import React, { useState, useEffect } from 'react';
import apiClient from '../../utils/api';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  AlertTitle
} from '@mui/material';

// Define interfaces for the expected data structure from the API
interface DashboardData {
  overall_status: {
    health: string;
    message: string;
  };
  quality_overview: any; // Define more specific types later
  realtime_metrics: any;
  ml_performance: any;
}

const AnalyticsDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await apiClient.get('/analytics/dashboard');
        setData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch analytics data. The backend service may be unavailable.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Optional: set up polling to refresh the data periodically
    const intervalId = setInterval(fetchData, 30000); // Refresh every 30 seconds

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, []);

  if (loading && !data) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading Dashboard...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        <AlertTitle>Error</AlertTitle>
        {error}
      </Alert>
    );
  }

  if (!data) {
    return <Typography>No analytics data available.</Typography>;
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Analytics Dashboard
      </Typography>
      
      {/* For now, just display the raw JSON to confirm data is being fetched */}
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </Paper>
  );
};

export default AnalyticsDashboard; 