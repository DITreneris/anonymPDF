import React, { Component, type ReactNode } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  AlertTitle,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Chip,
} from '@mui/material';
import {
  ErrorOutline,
  Refresh,
  BugReport,
  ExpandMore,
  Home,
} from '@mui/icons-material';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  errorId: string;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Generate a unique error ID for tracking
    const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    return {
      hasError: true,
      error,
      errorId,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Log error to console for development
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // In a real application, you might want to log this to an error reporting service
    this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error: Error, errorInfo: React.ErrorInfo) => {
    // This would typically send to an error monitoring service like Sentry
    const errorData = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      errorId: this.state.errorId,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    };

    // For now, just log to console
    console.error('Error logged:', errorData);
    
    // In production, you would send this to your error tracking service:
    // errorTrackingService.captureException(error, { extra: errorData });
  };

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, errorId } = this.state;

      return (
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.default',
            p: 3,
          }}
        >
          <Paper
            sx={{
              maxWidth: 600,
              width: '100%',
              p: 4,
              textAlign: 'center',
            }}
          >
            <ErrorOutline
              sx={{
                fontSize: 64,
                color: 'error.main',
                mb: 2,
              }}
            />
            
            <Typography variant="h4" gutterBottom color="error">
              Oops! Something went wrong
            </Typography>
            
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              We're sorry, but an unexpected error occurred. Don't worry - your data is safe.
            </Typography>

            <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
              <AlertTitle>Error Details</AlertTitle>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Error ID:</strong> {errorId}
              </Typography>
              <Typography variant="body2">
                <strong>Message:</strong> {error?.message || 'Unknown error'}
              </Typography>
            </Alert>

            <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 3 }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={this.handleRetry}
                color="primary"
              >
                Try Again
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Home />}
                onClick={this.handleGoHome}
              >
                Go Home
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={this.handleReload}
                color="secondary"
              >
                Reload Page
              </Button>
            </Stack>

            <Accordion sx={{ textAlign: 'left' }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <BugReport sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="subtitle2">
                    Technical Details (for developers)
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={2}>
                  <Box>
                    <Chip label="Error Stack" size="small" sx={{ mb: 1 }} />
                    <Typography
                      variant="body2"
                      component="pre"
                      sx={{
                        bgcolor: 'grey.100',
                        p: 2,
                        borderRadius: 1,
                        fontSize: '0.75rem',
                        overflow: 'auto',
                        maxHeight: 200,
                      }}
                    >
                      {error?.stack || 'No stack trace available'}
                    </Typography>
                  </Box>
                  
                  {errorInfo && (
                    <Box>
                      <Chip label="Component Stack" size="small" sx={{ mb: 1 }} />
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          bgcolor: 'grey.100',
                          p: 2,
                          borderRadius: 1,
                          fontSize: '0.75rem',
                          overflow: 'auto',
                          maxHeight: 200,
                        }}
                      >
                        {errorInfo.componentStack}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </AccordionDetails>
            </Accordion>

            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              If this problem persists, please contact support with Error ID: {errorId}
            </Typography>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 