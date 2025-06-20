import React from 'react';
import './InfoCard.css';

interface StatisticProps {
    label: string;
    value: string | number;
    tooltip?: string;
}

export const Statistic: React.FC<StatisticProps> = ({ label, value, tooltip }) => (
    <div className="statistic" title={tooltip}>
        <span className="statistic-label">{label}</span>
        <span className="statistic-value">{value}</span>
    </div>
);

interface InfoCardProps {
    title: string;
    children: React.ReactNode;
    isLoading?: boolean;
}

export const InfoCard: React.FC<InfoCardProps> = ({ title, children, isLoading = false }) => {
    return (
        <div className={`info-card ${isLoading ? 'loading' : ''}`}>
            <h3 className="info-card-title">{title}</h3>
            <div className="info-card-content">
                {children}
            </div>
        </div>
    );
}; 