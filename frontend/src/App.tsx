import React, { useCallback, useRef, useState, useEffect, Suspense, lazy } from 'react';
import apiClient from './utils/api';
import {
  Box,
  Container,
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Stack,
  Paper,
  Button,
  Tabs,
  Tab,
  CircularProgress,
} from '@mui/material';
import {
  CheckCircle,
} from '@mui/icons-material';
import { FRONTEND_VERSION } from './version';

// Import our new components
import TwoPaneLayout from './components/layout/TwoPaneLayout';
import FileUploadZone from './components/upload/FileUploadZone';
import ErrorBoundary from './components/ErrorBoundary';
import ErrorDialog, { type ProcessingError } from './components/ErrorDialog';
import { 
  createProcessingError,
  logError,
  isRetryableError 
} from './utils/errorHandler';
import RedactionReport, { type RedactionReport as RedactionReportData } from './components/RedactionReport';
const AnalyticsDashboard = lazy(() => import('./components/dashboard/AnalyticsDashboard'));

const API_BASE_URL = "/api/v1";
const PROCESSING_TIMEOUT = 30000; // 30 seconds

// Balanced theme with better contrast and visual weight
const theme = createTheme({
  palette: {
    primary: {
      main: '#005FCC',      // Polished cobalt blue - single brand accent
      light: '#4A90E2',
      dark: '#003D8A',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#6B7280',      // Neutral gray
      light: '#9CA3AF',
      dark: '#374151',
    },
    success: {
      main: '#059669',      // Slightly darker for better contrast
    },
    warning: {
      main: '#D97706',      // Slightly darker for better contrast
    },
    error: {
      main: '#DC2626',      // Slightly darker for better contrast
    },
    info: {
      main: '#2563EB',      // Slightly darker for better contrast
    },
    background: {
      default: '#F9FAFB',   // Slightly warmer neutral canvas
      paper: '#FFFFFF',     // White cards
    },
    text: {
      primary: '#111827',   // Darker for better readability
      secondary: '#6B7280', // Medium gray for help text
    },
    divider: '#D1D5DB',     // Slightly darker borders for definition
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { 
      fontSize: '32px', 
      fontWeight: 700,      // Bolder for better presence
      lineHeight: 1.2,
      color: '#111827',
    },
    h2: { 
      fontSize: '28px', 
      fontWeight: 600, 
      lineHeight: 1.3,
      color: '#111827',
    },
    h6: { 
      fontSize: '20px',     // Slightly smaller but still prominent
      fontWeight: 600,
      color: '#111827',
    },
    body1: { 
      fontSize: '16px',
      fontWeight: 400,
      lineHeight: 1.6,
      color: '#374151',     // Darker for better readability
    },
    body2: { 
      fontSize: '14px',
      fontWeight: 400,      // Slightly heavier
      lineHeight: 1.5,
      color: '#6B7280',
    },
    caption: {
      fontSize: '14px',
      fontWeight: 400,
      color: '#6B7280',
    },
  },
  spacing: 8, // 8px spacing scale
  shape: {
    borderRadius: 12, // Slightly more rounded for modern feel
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
          border: '1px solid #E5E7EB',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)', // Stronger shadow
          borderRadius: '12px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderRadius: '8px',
        },
      },
    },
  },
});

interface UploadResponse {
  id: number;
  original_filename: string;
  anonymized_filename?: string;
  status: string;
  redaction_report?: string;
  error_message?: string;
}

const App: React.FC = () => {
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [currentError, setCurrentError] = useState<ProcessingError | null>(null);
  const [showErrorDialog, setShowErrorDialog] = useState<boolean>(false);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [redactionReport, setRedactionReport] = useState<RedactionReportData | null>(null);
  const processingTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [backendVersion, setBackendVersion] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    // Cleanup polling on component unmount
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await apiClient.get('/pdf/version');
        if (response.data && response.data.version) {
          setBackendVersion(response.data.version);
        }
      } catch (error) {
        console.error("Could not fetch backend version:", error);
        setBackendVersion('N/A');
      }
    };

    fetchVersion();
  }, []);

  const handleError = useCallback((error: ProcessingError) => {
    setCurrentError(error);
    setShowErrorDialog(true);
    setUploadStatus('error');
    logError(error);
  }, []);

  const handleCloseErrorDialog = useCallback(() => {
    setShowErrorDialog(false);
    setCurrentError(null);
  }, []);

  const handleRetryOperation = useCallback(() => {
    setShowErrorDialog(false);
    setCurrentError(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setFileName(null);
    setUploadResponse(null);
    setRedactionReport(null);
  }, []);

  const pollForProcessingStatus = useCallback(async (documentId: number) => {
    if (pollIntervalRef.current) return; // Polling already active

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const response = await apiClient.get(`/pdf/documents/${documentId}`);
        const document = response.data;

        if (document.status === 'completed' || document.status === 'failed') {
          stopPolling();
          setUploadStatus(document.status === 'completed' ? 'success' : 'error');
          setUploadResponse(document);

          if (document.status === 'completed' && document.redaction_report) {
            try {
              const reportData = JSON.parse(document.redaction_report);
              setRedactionReport(reportData);
            } catch (error) {
              console.error('Failed to parse redaction report:', error);
              handleError(createProcessingError({
                message: 'Failed to parse the redaction report from the server.',
              }));
            }
          } else if (document.status === 'failed') {
            handleError(createProcessingError({
              response: { data: { detail: { message: document.error_message || 'Processing failed without a specific error.' } } }
            }));
          }
        }
      } catch (error) {
        stopPolling();
        handleError(createProcessingError(error));
      }
    }, 2000); // Poll every 2 seconds
  }, [handleError, stopPolling]);

  // File upload handler
  const handleFileAccepted = useCallback(async (file: File) => {
    setFileName(file.name);
    setUploadStatus('uploading');
    setUploadProgress(0);
    stopPolling(); // Stop any previous polling

    try {
      const formData = new FormData();
      formData.append('file', file);

      // No more onUploadProgress, as the file is just being accepted
      const response = await apiClient.post('/pdf/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: PROCESSING_TIMEOUT,
      });

      // On 202 Accepted, we start polling
      if (response.status === 202) {
        setUploadStatus('processing');
        const documentId = response.data.id;
        setUploadResponse(response.data); // Store initial response
        pollForProcessingStatus(documentId);
      } else {
        // This case should ideally not happen with the new backend
        handleError(createProcessingError({
          message: `Unexpected status code: ${response.status}`
        }));
      }
    } catch (error) {
      handleError(createProcessingError(error));
    }
  }, [handleError, pollForProcessingStatus, stopPolling]);

  const handleFileRejected = useCallback((rejections: any[]) => {
    if (rejections.length > 0) {
      const rejection = rejections[0];
      const errorMessage = rejection.errors?.[0]?.message || 'File upload failed';
      
      const error: ProcessingError = {
        type: 'validation',
        message: 'File Upload Error',
        details: errorMessage,
        timestamp: new Date().toISOString(),
        recoveryActions: [
          'Ensure your file is a valid PDF',
          'Check that the file is not corrupted',
          'Try with a different PDF file'
        ]
      };
      
      handleError(error);
    }
  }, [handleError]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (processingTimeout.current) clearTimeout(processingTimeout.current);
  }, []);

  // Left Pane Content - Now with actual upload functionality
  const leftPaneContent = (
    <Stack spacing={4} sx={{ height: '100%' }}>
      <Box>
        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', mb: 1, fontWeight: 600 }}>
          Upload Document
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Select a PDF file to anonymize personal information
        </Typography>
      </Box>

      <FileUploadZone
        onFileAccepted={handleFileAccepted}
        onFileRejected={handleFileRejected}
        currentFileName={fileName}
        isProcessing={uploadStatus === 'uploading' || uploadStatus === 'processing'}
        disabled={uploadStatus === 'uploading' || uploadStatus === 'processing'}
        maxSize={50 * 1024 * 1024} // 50MB
      />

      <Paper 
        sx={{ 
          p: 3,
          border: '1px solid #E5E7EB',
          borderRadius: '12px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          transition: 'all 0.2s ease',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
          },
        }}
      >
        <Box display="flex" alignItems="center" gap={3} mb={2}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: '10px',
              backgroundColor: uploadStatus === 'success' ? 'success.main' : 
                              uploadStatus === 'error' ? 'error.main' :
                              uploadStatus === 'uploading' || uploadStatus === 'processing' ? 'warning.main' :
                              'success.main',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 4px 6px ${theme.palette.success.main}33`,
            }}
          >
            <CheckCircle sx={{ fontSize: 20, color: 'white' }} />
          </Box>
          <Typography variant="h6" color="text.primary" sx={{ fontWeight: 600 }}>
            {uploadStatus === 'success' ? 'Processing Complete' :
             uploadStatus === 'error' ? 'Upload Error' :
             uploadStatus === 'uploading' ? 'Uploading...' :
             uploadStatus === 'processing' ? 'Processing...' :
             'Ready to Process'}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {uploadStatus === 'success' ? 'Your document has been successfully processed' :
           uploadStatus === 'error' ? 'There was an error processing your document' :
           uploadStatus === 'uploading' ? `Uploading ${fileName}... ${uploadProgress}%` :
           uploadStatus === 'processing' ? 'Analyzing and redacting personal information...' :
           'Upload a PDF file to begin anonymization'}
        </Typography>
        
        {uploadStatus === 'success' && (
          <Box sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              onClick={handleRetryOperation}
              sx={{ 
                borderRadius: '8px',
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              Process Another File
            </Button>
          </Box>
        )}
      </Paper>
    </Stack>
  );

  // Right Pane Content - Combined Statistics and How It Works
  const rightPaneContent = (
    !uploadResponse ? (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6">Welcome</Typography>
        <Typography>Upload a document to get started.</Typography>
      </Paper>
    ) : (
      <RedactionReport
        documentId={uploadResponse.id}
        report={redactionReport}
        downloadUrl={uploadResponse.anonymized_filename ? `${API_BASE_URL}/pdf/download/${uploadResponse.anonymized_filename}` : undefined}
      />
    )
  );

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            minHeight: '100vh',
            backgroundColor: 'background.default',
            py: 5,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Container maxWidth="xl" sx={{ flex: 1 }}>
            <Typography 
              variant="h1" 
              component="h1" 
              gutterBottom 
              align="center"
              sx={{ 
                mb: 5,
                color: 'primary.main',
                fontWeight: 700,
              }}
            >
              AnonymPDF
            </Typography>
            
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
              <Tabs value={activeTab} onChange={handleTabChange} centered>
                <Tab label="Anonymize PDF" />
                <Tab label="Analytics Dashboard" />
              </Tabs>
            </Box>

            {activeTab === 0 && (
              <TwoPaneLayout
                leftPane={leftPaneContent}
                rightPane={rightPaneContent}
                leftPaneWidth={45}
                spacing={4}
              />
            )}
            {activeTab === 1 && (
              <Suspense fallback={
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
                  <CircularProgress />
                </Box>
              }>
                <AnalyticsDashboard />
              </Suspense>
            )}

          </Container>
          <Box sx={{ textAlign: 'center', color: 'text.secondary', fontSize: 13, py: 1, borderTop: '1px solid #E5E7EB', mt: 4 }}>
            AnonymPDF v{FRONTEND_VERSION} (frontend)
            {backendVersion && (
              <span> | API v{backendVersion} (backend)</span>
            )}
          </Box>

          {/* Enhanced Error Dialog */}
          <ErrorDialog
            open={showErrorDialog}
            error={currentError}
            onClose={handleCloseErrorDialog}
            onRetry={currentError && isRetryableError(currentError) ? handleRetryOperation : undefined}
          />
        </Box>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default App;