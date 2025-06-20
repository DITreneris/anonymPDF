import React, { useState, useEffect } from 'react';
import { getMonitoringStatus } from '../../utils/api';
import { InfoCard, Statistic } from '../ui/InfoCard';
import './StatusDashboard.css';

const StatusDashboard: React.FC = () => {
    const [status, setStatus] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                // Don't set loading to true on refetch, only on initial load
                if (!status) setIsLoading(true);
                const data = await getMonitoringStatus();
                setStatus(data);
                setError(null);
            } catch (err) {
                setError('Failed to fetch monitoring status. The backend may be busy or down.');
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchStatus();
        const intervalId = setInterval(fetchStatus, 5000); // Poll every 5 seconds

        return () => clearInterval(intervalId);
    }, [status]);

    const renderSkeletons = () => (
        <InfoCard title="System Status" isLoading={true}>
            <Statistic label="Avg. Processing Time" value="..." />
            <Statistic label="Avg. CPU Usage" value="..." />
            <Statistic label="Avg. Memory Usage" value="..." />
            <Statistic label="Latest CPU" value="..." />
            <Statistic label="Latest Memory" value="..." />
        </InfoCard>
    );

    if (error) {
        return <InfoCard title="System Status"><div className="error-message">{error}</div></InfoCard>;
    }
    
    if (isLoading && !status) {
        return renderSkeletons();
    }

    if (!status || status.status === 'No data available.') {
        return <InfoCard title="System Status"><p>No monitoring data available yet. Please process a document to see live stats.</p></InfoCard>;
    }

    return (
        <InfoCard title="System Status" isLoading={isLoading && !status}>
            <Statistic 
                label="Avg. Processing Time" 
                value={`${status.average_duration_ms} ms`} 
                tooltip="Average time taken per document processed in the last 500 events."
            />
            <Statistic 
                label="Avg. CPU Usage" 
                value={`${status.average_cpu_percent}%`}
                tooltip="Average CPU utilization of the worker process over the last 500 events."
            />
            <Statistic 
                label="Avg. Memory Usage" 
                value={`${status.average_memory_mb} MB`}
                tooltip="Average memory consumption of the worker process over the last 500 events."
            />
             <Statistic 
                label="Latest CPU" 
                value={`${status.latest_cpu_percent}%`}
                tooltip="Most recently recorded CPU utilization of the worker process."
            />
            <Statistic 
                label="Latest Memory" 
                value={`${status.latest_memory_mb} MB`}
                tooltip="Most recently recorded memory consumption of the worker process."
            />
            <div className="status-footer">
                <span>Total Events: {status.total_events}</span>
                <span className='last-updated'>Last Event: {status.latest_event_timestamp}</span>
            </div>
        </InfoCard>
    );
};

export default StatusDashboard; 