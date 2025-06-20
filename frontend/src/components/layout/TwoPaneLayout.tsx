import React from 'react';
import StatusDashboard from './StatusDashboard';
import './TwoPaneLayout.css';

interface TwoPaneLayoutProps {
  leftPane: React.ReactNode;
  rightPane: React.ReactNode;
  leftPaneWidth?: number; 
  spacing?: number; 
  elevation?: number; 
  className?: string;
  leftVisible?: boolean;
}

const TwoPaneLayout: React.FC<TwoPaneLayoutProps> = ({
  leftPane,
  rightPane,
  className,
  leftVisible = true,
}) => {

  return (
    <div className={`two-pane-layout ${className || ''}`}>
      <div className={`left-pane ${leftVisible ? 'visible' : ''}`}>
        <div className="left-pane-content">
          {leftPane}
        </div>
        <div className="status-dashboard-container">
          <StatusDashboard />
        </div>
      </div>
      <div className="right-pane">
        {rightPane}
      </div>
    </div>
  );
};

export default TwoPaneLayout; 