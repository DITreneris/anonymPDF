import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import StatusDashboard from './StatusDashboard';
import { getMonitoringStatus } from '../../utils/api';

// Mock the API utility
jest.mock('../../utils/api', () => ({
  getMonitoringStatus: jest.fn(),
}));

// Cast the mock to the correct type to allow mockResolvedValue etc.
const mockedGetMonitoringStatus = getMonitoringStatus as jest.Mock;

describe('StatusDashboard', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders loading skeletons initially', () => {
    mockedGetMonitoringStatus.mockImplementation(() => new Promise(() => {})); // Never resolves
    render(<StatusDashboard />);
    
    expect(screen.getByText('Avg. Processing Time')).toBeInTheDocument();
    expect(screen.getAllByText('...')).toHaveLength(5);
  });

  it('displays an error message when the API call fails', async () => {
    const errorMessage = 'Failed to fetch';
    mockedGetMonitoringStatus.mockRejectedValue(new Error(errorMessage));
    render(<StatusDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/failed to fetch monitoring status/i)).toBeInTheDocument();
    });
  });

  it('displays "No data available" message when API returns no data', async () => {
    mockedGetMonitoringStatus.mockResolvedValue({ status: 'No data available.' });
    render(<StatusDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/no monitoring data available yet/i)).toBeInTheDocument();
    });
  });

  it('displays the status data when the API call is successful', async () => {
    const mockData = {
      average_duration_ms: 123.45,
      average_cpu_percent: 25.5,
      average_memory_mb: 150.8,
      latest_cpu_percent: 30.1,
      latest_memory_mb: 155.2,
      total_events: 42,
      latest_event_timestamp: 'Tue Jun 18 2024 10:00:00',
    };
    mockedGetMonitoringStatus.mockResolvedValue(mockData);
    render(<StatusDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Avg. Processing Time')).toBeInTheDocument();
      expect(screen.getByText('123.45 ms')).toBeInTheDocument();
      expect(screen.getByText('25.5%')).toBeInTheDocument();
      expect(screen.getByText('150.8 MB')).toBeInTheDocument();
      expect(screen.getByText('30.1%')).toBeInTheDocument();
      expect(screen.getByText('155.2 MB')).toBeInTheDocument();
      expect(screen.getByText('Total Events: 42')).toBeInTheDocument();
      expect(screen.getByText(/Last Event:.*10:00:00/)).toBeInTheDocument();
    });
  });
}); 