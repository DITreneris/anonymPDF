import React, { useCallback, useRef, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  Box,
  Container,
  Paper,
  Typography,
  LinearProgress,
  Alert,
  AlertTitle,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Grid,
  CircularProgress,
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import RedactionReport from './components/RedactionReport';

const BACKEND_URL = "http://127.0.0.1:8000/api/v1/upload";
const PROCESSING_TIMEOUT = 30000; // 30 seconds

// Create a theme instance with custom colors and shadows
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
        },
      },
    },
  },
});

interface UploadResponse {
  status: string;
  redactionReport?: RedactionReportData;
  downloadUrl?: string;
  uploadId?: string;
  id?: number;
  error_message?: string;
  original_filename?: string;
}

interface RedactionReportData {
  title: string;
  detectedLanguage: string;
  totalRedactions: number;
  categories: { [key: string]: number };
  details?: any;
  error?: string;
  rawReportString?: string;
}

const App: React.FC = () => {
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const processingTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleExportReport = useCallback(() => {
    if (uploadResponse?.redactionReport) {
      const reportData = JSON.stringify(uploadResponse.redactionReport, null, 2);
      const blob = new Blob([reportData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `redaction-report-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }, [uploadResponse]);

  const pollProcessingStatus = useCallback(async (uploadId: string) => {
    try {
      interface DocumentStatusResponse {
        id: number;
        original_filename: string;
        anonymized_filename?: string;
        file_size: number;
        status: string;
        created_at: string;
        updated_at: string;
        error_message?: string;
        download_url?: string;
      }

      const response = await axios.get<DocumentStatusResponse>(`http://127.0.0.1:8000/api/v1/documents/${uploadId}`, {
        headers: { 'Accept': 'application/json' }
      });

      if (response.data.status === 'completed') {
        let parsedReport: RedactionReportData | undefined;
        let absDownloadUrl: string | undefined;

        if (response.data.download_url) {
          const apiUrlBase = BACKEND_URL.substring(0, BACKEND_URL.indexOf('/api/v1'));
          absDownloadUrl = apiUrlBase + response.data.download_url;
        } else if (response.data.anonymized_filename) {
          const apiUrlBase = BACKEND_URL.substring(0, BACKEND_URL.indexOf('/api/v1'));
          absDownloadUrl = `${apiUrlBase}/api/v1/download/${response.data.anonymized_filename}`;
        }

        if (response.data.error_message) {
          try {
            const reportObj = JSON.parse(response.data.error_message);
            if (reportObj && typeof reportObj.totalRedactions === 'number' && reportObj.categories) {
              parsedReport = reportObj as RedactionReportData;
            } else {
              parsedReport = {
                title: "Processing Error",
                detectedLanguage: "N/A",
                totalRedactions: 0,
                categories: {},
                error: "Invalid report format received.",
                rawReportString: response.data.error_message
              };
            }
          } catch (e) {
            console.error("Failed to parse redaction report JSON:", e);
            parsedReport = {
              title: "Redaction Report (Raw)",
              detectedLanguage: "N/A",
              totalRedactions: 0,
              categories: {},
              error: "Report is not valid JSON.",
              rawReportString: response.data.error_message
            };
          }
        }

        setUploadResponse({
          status: response.data.status,
          redactionReport: parsedReport,
          downloadUrl: absDownloadUrl,
          uploadId: String(response.data.id),
          id: response.data.id,
          original_filename: response.data.original_filename,
          error_message: response.data.error_message
        });
        setUploadStatus('success');
        if (processingTimeout.current) clearTimeout(processingTimeout.current);
        return true;
      }
      if (response.data.status === 'pending' || response.data.status === 'processing') {
        return false;
      }
      if (response.data.status === 'failed') {
        let errorMessage = "Processing failed on the server.";
        if (response.data.error_message) {
          try {
            const errorObj = JSON.parse(response.data.error_message);
            errorMessage = errorObj.error || errorObj.details || response.data.error_message;
          } catch (e) {
            errorMessage = response.data.error_message;
          }
        }
        setErrorMessage(errorMessage);
        setUploadStatus('error');
        if (processingTimeout.current) clearTimeout(processingTimeout.current);
        return true;
      }
      return false;
    } catch (err) {
      console.error("Polling error:", err);
      return false;
    }
  }, []);

  const startPolling = useCallback(async (uploadId: string) => {
    let attempts = 0;
    const maxAttempts = 30; // 30 seconds

    const poll = async () => {
      if (attempts >= maxAttempts) {
        setUploadStatus('error');
        setErrorMessage('Processing is taking longer than expected. Please try again or contact support.');
        return;
      }
      const done = await pollProcessingStatus(uploadId);
      if (!done) {
        attempts++;
        setTimeout(poll, 1000);
      }
    };

    await poll();
  }, [pollProcessingStatus]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return;
    const file = acceptedFiles[0];

    setFileName(file.name);
    setUploadStatus('uploading');
    setErrorMessage(null);
    setUploadProgress(0);

    if (processingTimeout.current) {
      clearTimeout(processingTimeout.current);
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const uploadResp = await axios.post<UploadResponse>(BACKEND_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Accept': 'application/json'
        },
        onUploadProgress: (e) => {
          const percent = e.total ? Math.round((e.loaded * 100) / e.total) : 0;
          setUploadProgress(percent);
        },
      });

      console.log('↪ upload response:', uploadResp);
      console.log('↪ response.data:', uploadResp.data);

      setUploadStatus('processing');

      const documentId = uploadResp.data.id;

      if (documentId) {
        processingTimeout.current = setTimeout(() => {
          if (uploadStatus !== 'success' && uploadStatus !== 'error') {
            setUploadStatus('error');
            setErrorMessage('Processing is taking longer than expected. Please try again or contact support.');
          }
        }, PROCESSING_TIMEOUT);
        await startPolling(String(documentId));
      } else {
        console.error('No document ID (uploadResp.data.id) received from POST /upload response', uploadResp.data);
        throw new Error('No upload ID received from server');
      }

    } catch (err: any) {
      console.error('Upload error:', err);
      setUploadStatus('error');
      setErrorMessage(
        err.response?.data?.detail ||
        err.message ||
        'An error occurred during upload. Please try again.'
      );
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => () => {
    if (processingTimeout.current) clearTimeout(processingTimeout.current);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(45deg, #f3f4f6 0%, #fff 100%)',
          py: 4,
        }}
      >
        <Container maxWidth="md">
          <Typography 
            variant="h2" 
            component="h1" 
            gutterBottom 
            align="center"
            sx={{ 
              mb: 4,
              fontWeight: 'bold',
              background: 'linear-gradient(45deg, #1976d2 30%, #2196f3 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            AnonymPDF
          </Typography>
          
          <Box display="flex" flexDirection="column" alignItems="center" width="100%">
            <Box width="100%" maxWidth={600}>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 6,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  cursor: 'pointer',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  borderRadius: 2,
                  transition: 'all 0.3s ease',
                  maxWidth: 600,
                  width: '100%',
                  '&:hover': {
                    bgcolor: 'action.hover',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                  },
                }}
              >
                <input {...getInputProps()} />
                <CloudUpload 
                  sx={{ 
                    fontSize: 64, 
                    color: 'primary.main',
                    mb: 2,
                    opacity: isDragActive ? 0.8 : 1,
                  }} 
                />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop the PDF here' : 'Drag & drop your PDF'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to select a file
                </Typography>
                {fileName && (
                  <Typography variant="body2" sx={{ mt: 2, color: 'primary.main' }}>
                    Selected: {fileName}
                  </Typography>
                )}
              </Paper>

              {(uploadStatus === 'uploading' || uploadStatus === 'processing') && (
                <Paper sx={{ mt: 2, p: 2 }}>
                  <Box sx={{ width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <CircularProgress size={20} sx={{ mr: 1 }} />
                      <Typography variant="body2" color="text.secondary">
                        {uploadStatus === 'uploading' ? 'Uploading...' : 'Processing document...'}
                      </Typography>
                    </Box>
                    {uploadStatus === 'uploading' && (
                      <LinearProgress variant="determinate" value={uploadProgress} sx={{ height: 8, borderRadius: 4 }} />
                    )}
                    {uploadStatus === 'processing' && (
                      <LinearProgress sx={{ height: 8, borderRadius: 4 }} />
                    )}
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      {uploadStatus === 'uploading' ? `${uploadProgress}% uploaded` : 'This may take a few moments...'}
                    </Typography>
                  </Box>
                </Paper>
              )}

              {uploadStatus === 'error' && (
                <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }}>
                  <AlertTitle>Error</AlertTitle>
                  {errorMessage}
                </Alert>
              )}
            </Box>

            {uploadStatus === 'success' && uploadResponse && (
              <Box mt={4} width="100%" maxWidth={700}>
                <RedactionReport
                  report={uploadResponse.redactionReport}
                  downloadUrl={uploadResponse.downloadUrl}
                  onExportReport={handleExportReport}
                />
              </Box>
            )}
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App;