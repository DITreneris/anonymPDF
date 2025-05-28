import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Button,
  Box,
  Alert,
  AlertTitle,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Stack,
  IconButton,
} from '@mui/material';
import {
  ErrorOutline,
  ExpandMore,
  CheckCircleOutline,
  Close,
  ContentCopy,
  Refresh,
  Warning,
  Info,
} from '@mui/icons-material';

export interface ProcessingError {
  type: 'validation' | 'processing' | 'system' | 'network' | 'timeout';
  message: string;
  details?: string;
  recoveryActions?: string[];
  errorCode?: string;
  timestamp?: string;
  context?: Record<string, any>;
}

interface ErrorDialogProps {
  open: boolean;
  error: ProcessingError | null;
  onClose: () => void;
  onRetry?: () => void;
  title?: string;
}

const ErrorDialog: React.FC<ErrorDialogProps> = ({
  open,
  error,
  onClose,
  onRetry,
  title = "Processing Error"
}) => {
  if (!error) return null;

  const getSeverityIcon = (type: ProcessingError['type']) => {
    switch (type) {
      case 'validation':
        return <Warning color="warning" />;
      case 'processing':
        return <ErrorOutline color="error" />;
      case 'system':
        return <ErrorOutline color="error" />;
      case 'network':
        return <Warning color="warning" />;
      case 'timeout':
        return <Info color="info" />;
      default:
        return <ErrorOutline color="error" />;
    }
  };

  const getSeverityColor = (type: ProcessingError['type']): 'error' | 'warning' | 'info' => {
    switch (type) {
      case 'validation':
      case 'network':
        return 'warning';
      case 'timeout':
        return 'info';
      default:
        return 'error';
    }
  };

  const getDefaultRecoveryActions = (type: ProcessingError['type']): string[] => {
    switch (type) {
      case 'validation':
        return [
          'Check that your file is a valid PDF',
          'Ensure the file is not corrupted',
          'Try with a different PDF file'
        ];
      case 'processing':
        return [
          'Try uploading the file again',
          'Check if the PDF contains readable text',
          'Contact support if the problem persists'
        ];
      case 'system':
        return [
          'Wait a moment and try again',
          'Refresh the page',
          'Contact support if the issue continues'
        ];
      case 'network':
        return [
          'Check your internet connection',
          'Try again in a few moments',
          'Refresh the page if the problem persists'
        ];
      case 'timeout':
        return [
          'The file may be large - try again',
          'Check your internet connection',
          'Contact support for very large files'
        ];
      default:
        return [
          'Try the operation again',
          'Refresh the page',
          'Contact support if needed'
        ];
    }
  };

  const handleCopyError = () => {
    const errorInfo = {
      type: error.type,
      message: error.message,
      details: error.details,
      errorCode: error.errorCode,
      timestamp: error.timestamp || new Date().toISOString(),
      context: error.context,
    };

    navigator.clipboard.writeText(JSON.stringify(errorInfo, null, 2));
  };

  const recoveryActions = error.recoveryActions || getDefaultRecoveryActions(error.type);

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 2 }
      }}
    >
      <DialogTitle>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center">
            {getSeverityIcon(error.type)}
            <Typography variant="h6" sx={{ ml: 1 }}>
              {title}
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Stack spacing={3}>
          {/* Main Error Alert */}
          <Alert severity={getSeverityColor(error.type)}>
            <AlertTitle>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <span>{error.message}</span>
                <Chip 
                  label={error.type.toUpperCase()} 
                  size="small" 
                  variant="outlined"
                  color={getSeverityColor(error.type)}
                />
              </Box>
            </AlertTitle>
            {error.details && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                {error.details}
              </Typography>
            )}
            {error.errorCode && (
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Error Code: {error.errorCode}
              </Typography>
            )}
          </Alert>

          {/* Recovery Actions */}
          {recoveryActions.length > 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Suggested Actions:
              </Typography>
              <List dense>
                {recoveryActions.map((action, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <CheckCircleOutline color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={action}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {/* Technical Details (Collapsible) */}
          {(error.context || error.timestamp) && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2">
                  Technical Details
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={2}>
                  {error.timestamp && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Timestamp:
                      </Typography>
                      <Typography variant="body2">
                        {new Date(error.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                  )}
                  
                  {error.context && (
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Context:
                      </Typography>
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          bgcolor: 'grey.100',
                          p: 1,
                          borderRadius: 1,
                          fontSize: '0.75rem',
                          overflow: 'auto',
                          maxHeight: 150,
                        }}
                      >
                        {JSON.stringify(error.context, null, 2)}
                      </Typography>
                    </Box>
                  )}

                  <Button
                    startIcon={<ContentCopy />}
                    onClick={handleCopyError}
                    size="small"
                    variant="outlined"
                  >
                    Copy Error Details
                  </Button>
                </Stack>
              </AccordionDetails>
            </Accordion>
          )}
        </Stack>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} color="inherit">
          Close
        </Button>
        {onRetry && (
          <Button
            onClick={onRetry}
            variant="contained"
            startIcon={<Refresh />}
            color="primary"
          >
            Try Again
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default ErrorDialog; 