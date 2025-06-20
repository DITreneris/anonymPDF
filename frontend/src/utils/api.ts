import axios from 'axios';
import axiosRetry from 'axios-retry';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Create a new axios instance
const apiClient = axios.create({
  baseURL: API_URL,
});

// Configure axios-retry
axiosRetry(apiClient, {
  retries: 3, // Number of retries
  retryDelay: (retryCount) => {
    console.log(`Retrying request, attempt number ${retryCount}`);
    return retryCount * 1000; // Exponential back-off: 1s, 2s, 3s
  },
  retryCondition: (error) => {
    // Retry on network errors and 5xx server errors
    return (
      axiosRetry.isNetworkError(error) ||
      (error.response ? error.response.status >= 500 : false)
    );
  },
});

export default apiClient;

// Define interfaces based on backend models
interface FeedbackItem {
  text_segment: string;
  original_category: string;
  is_correct: boolean;
}

export interface FeedbackPayload {
  document_id: string;
  feedback_items: FeedbackItem[];
}

export const submitFeedback = async (feedbackData: FeedbackPayload): Promise<{ message: string }> => {
  const response = await apiClient.post('/feedback', feedbackData);
  return response.data;
};

export const getMonitoringStatus = async (): Promise<any> => {
  const response = await apiClient.get('/monitoring/status');
  return response.data;
}; 